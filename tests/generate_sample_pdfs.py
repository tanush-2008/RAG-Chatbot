"""One-off script that generates the sample PDF documents used for demos and testing.

Run with: python tests/generate_sample_pdfs.py
Requires reportlab (dev-only dependency, not needed to run the app itself).
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUT_DIR = Path(__file__).resolve().parent.parent / "documents"
styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"])
h2 = ParagraphStyle("H2", parent=styles["Heading2"])
body = ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=10, leading=15)


def build_pdf(filename: str, title: str, sections: list[tuple[str, str]]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [Paragraph(title, h1), Spacer(1, 0.3 * inch)]
    for i, (heading, text) in enumerate(sections):
        story.append(Paragraph(heading, h2))
        story.append(Paragraph(text, body))
        if i < len(sections) - 1:
            story.append(PageBreak())
    doc.build(story)
    print(f"Wrote {path}")


POLICY_SECTIONS = [
    ("1. Purpose", "This Leave Policy explains the types of leave available to employees of "
     "Northwind Retail Pvt. Ltd., how leave is accrued, and the process for requesting time off. "
     "It applies to all full-time employees from their date of joining."),
    ("2. Types of Leave", "Employees are entitled to the following categories of leave each "
     "calendar year: 12 days of Casual Leave, 12 days of Sick Leave, 15 days of Earned Leave, "
     "and 5 days of Paid Bereavement Leave. Casual Leave cannot be carried forward to the next "
     "year. Earned Leave can be carried forward up to a maximum of 45 days."),
    ("3. Leave Accrual", "Earned Leave accrues at the rate of 1.25 days per completed month of "
     "service. Casual Leave and Sick Leave are credited in full at the start of the calendar "
     "year. New employees joining mid-year receive leave on a pro-rata basis."),
    ("4. Applying for Leave", "Employees must apply for leave through the HR portal at least "
     "3 working days in advance for planned leave. Sick Leave taken without prior notice must be "
     "informed to the reporting manager on the same day and regularized within 3 working days of "
     "returning to work. Leave requests longer than 5 consecutive days require approval from both "
     "the reporting manager and the department head."),
    ("5. Unpaid Leave", "Employees who have exhausted their paid leave balance may request "
     "Leave Without Pay (LWOP) for a maximum of 30 days per year, subject to manager approval. "
     "LWOP does not count towards years of service for benefits calculation."),
    ("6. Maternity and Paternity Leave", "Female employees are entitled to 26 weeks of paid "
     "Maternity Leave as per statutory requirements. Male employees are entitled to 10 working "
     "days of Paternity Leave, to be availed within 3 months of the child's birth."),
    ("7. Public Holidays", "The company observes 10 public holidays each year as per the list "
     "published annually by HR. Employees required to work on a public holiday are entitled to a "
     "compensatory day off within the following 30 days."),
    ("8. Leave Encashment", "Unused Earned Leave beyond the carry-forward limit of 45 days is "
     "automatically encashed at the end of the calendar year at the employee's basic daily pay "
     "rate."),
]

HANDBOOK_SECTIONS = [
    ("1. Welcome", "This Employee Handbook describes the policies, expectations, and benefits "
     "that apply to all employees of Northwind Retail Pvt. Ltd. Please read it carefully and "
     "reach out to Human Resources with any questions."),
    ("2. Working Hours", "Standard working hours are 9:30 AM to 6:30 PM, Monday through Friday, "
     "with a one-hour lunch break. Employees are expected to be available during core hours of "
     "11:00 AM to 4:00 PM for meetings and collaboration."),
    ("3. Attendance Calculation", "Attendance is calculated based on biometric check-in and "
     "check-out times recorded at the office entrance, or through the mobile app for remote "
     "days. An employee is marked 'Present' for a day if they log at least 4 hours of work "
     "between 9:00 AM and 8:00 PM. Logging between 4 and 6 hours counts as 'Half Day'. Less than "
     "4 hours, or no log at all without approved leave, is marked as 'Absent'. Monthly attendance "
     "percentage is calculated as (Present days + 0.5 x Half Days) divided by total working days "
     "in the month, multiplied by 100."),
    ("4. Remote Work Policy", "Employees may work remotely up to 2 days per week with prior "
     "manager approval logged in the HR portal. Fully remote arrangements require department "
     "head sign-off and are reviewed quarterly."),
    ("5. Code of Conduct", "All employees are expected to treat colleagues, clients, and vendors "
     "with respect and professionalism. Harassment, discrimination, or retaliation of any kind "
     "will not be tolerated and should be reported immediately to HR or through the anonymous "
     "ethics hotline."),
    ("6. IT and Data Security", "Employees must use company-issued devices for accessing "
     "internal systems, enable full-disk encryption, and never share login credentials. Any lost "
     "or stolen device must be reported to IT Security within 24 hours."),
    ("7. Expense Reimbursement", "Business-related expenses (travel, client meals, approved "
     "software subscriptions) are reimbursed within 15 working days of submitting an itemized "
     "claim with receipts through the Finance portal. Claims submitted more than 60 days after "
     "the expense was incurred will not be reimbursed."),
    ("8. Performance Reviews", "Formal performance reviews are conducted twice a year, in April "
     "and October. Reviews include a self-assessment, manager feedback, and goal-setting for the "
     "next cycle. Compensation revisions are typically communicated after the April review."),
    ("9. Grievance Redressal", "Employees with a workplace grievance should first raise it with "
     "their reporting manager. If unresolved within 10 working days, the matter can be escalated "
     "to HR, and finally to the Grievance Committee, which meets monthly."),
    ("10. Exit Process", "Employees resigning must serve a notice period of 60 days for "
     "non-managerial roles and 90 days for managerial roles. A full and final settlement, "
     "including any pending reimbursements and leave encashment, is processed within 45 days of "
     "the last working day."),
]


if __name__ == "__main__":
    build_pdf("Policy.pdf", "Northwind Retail Pvt. Ltd. - Leave Policy", POLICY_SECTIONS)
    build_pdf("Handbook.pdf", "Northwind Retail Pvt. Ltd. - Employee Handbook", HANDBOOK_SECTIONS)
