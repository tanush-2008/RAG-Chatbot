"""Module 6 (part 2): LLM provider abstraction.

Supports Groq, Gemini, and OpenAI through a single call_llm() function selected
by the LLM_PROVIDER environment variable. If no provider/API key is configured,
falls back to a local "extractive" mode so the app still runs end-to-end
without any paid API key (useful for offline grading/demo).
"""

from __future__ import annotations

import os


class LLMError(RuntimeError):
    pass


def _call_groq(messages: list[dict], model: str, api_key: str) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def _call_openai(messages: list[dict], model: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(messages: list[dict], model: str, api_key: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1),
    )
    return response.text.strip()


def _extractive_fallback(messages: list[dict]) -> str:
    """No LLM configured: return the retrieved context verbatim as a best-effort answer.

    This keeps Module 6 functional without any API key so the rest of the RAG
    pipeline (retrieval, chunking, sourcing) can still be demoed end-to-end.
    """
    user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "(no relevant context was retrieved)" in user_message:
        return "I could not find this information in the uploaded documents."
    context_start = user_message.find("Context:\n")
    question_start = user_message.find("\n\nQuestion:")
    context = user_message[context_start + len("Context:\n"): question_start].strip()
    return (
        "[No LLM API key configured - showing the most relevant retrieved passage "
        "instead of a generated answer]\n\n" + context
    )


PROVIDER_CALLERS = {
    "groq": (_call_groq, "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openai": (_call_openai, "OPENAI_API_KEY", "gpt-4o-mini"),
    "gemini": (_call_gemini, "GEMINI_API_KEY", "gemini-2.0-flash"),
}


def call_llm(messages: list[dict], provider: str | None = None, model: str | None = None) -> str:
    """Route a chat request to the configured provider, with a safe offline fallback."""
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()

    if provider not in PROVIDER_CALLERS:
        return _extractive_fallback(messages)

    caller, key_env_var, default_model = PROVIDER_CALLERS[provider]
    api_key = os.getenv(key_env_var)
    if not api_key:
        return _extractive_fallback(messages)

    resolved_model = model or os.getenv("LLM_MODEL") or default_model
    try:
        return caller(messages, resolved_model, api_key)
    except Exception as exc:
        raise LLMError(f"{provider} request failed: {exc}") from exc


def is_llm_configured(provider: str | None = None) -> bool:
    """Whether a real LLM (not the extractive fallback) is available to call."""
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    if provider not in PROVIDER_CALLERS:
        return False
    _, key_env_var, _ = PROVIDER_CALLERS[provider]
    return bool(os.getenv(key_env_var))


def condense_question(
    question: str,
    history: list[tuple[str, str]],
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Rewrite a follow-up question into a standalone one using recent chat history.

    Used only to build a better retrieval query; the original question (plus the
    raw history) is still what's sent to the LLM for the final answer, so the
    conversational tone of the reply is unaffected. Falls back to the original
    question unchanged if no LLM is configured (offline/extractive mode).
    """
    if not history or not is_llm_configured(provider):
        return question

    history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the follow-up question as a fully standalone question "
                "that includes any context implied by the conversation history "
                "(e.g. resolve pronouns like 'it' or 'that'). Return ONLY the "
                "rewritten question, with no explanation or quotation marks."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation history:\n{history_text}\n\nFollow-up question: {question}",
        },
    ]
    try:
        rewritten = call_llm(messages, provider=provider, model=model).strip().strip('"')
        return rewritten or question
    except LLMError:
        return question
