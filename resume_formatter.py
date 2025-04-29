# === resume_formatter.py ===

from typing import List
import re


def format_experience_section(jobs: List[dict]) -> str:
    """
    Formats a list of enhanced job dicts into a clean, copy-pasteable
    'Experience' section string with proper bullets and spacing.
    """
    lines = []

    for job in jobs:
        # Extract job components
        title = job.get('title', '').strip()
        company = job.get('company', '').strip()
        date_range = job.get('date_range', '').strip()
        
        # Format job title and date (first line)
        # Use diamond bullet and ensure proper spacing
        if date_range:
            job_header = f"◆ {title} {date_range}"
        else:
            job_header = f"◆ {title}"
        
        lines.append(job_header)
        
        # Add company on its own line if present (second line)
        if company:
            lines.append(company)
        
        # Add a blank line after the company/header for spacing
        lines.append("")
        
        # Format bullet points
        for bullet in job.get("bullets", []):
            # Clean up bullet formatting to ensure consistent bullets
            clean_bullet = bullet.lstrip('•-*□■◆♦📌 \t').strip()
            lines.append(f"• {clean_bullet}")
        
        # Add blank line between job entries
        lines.append("")

    result = "\n".join(lines).strip()
    return result


def assemble_resume(summary: str, skills: str, experience: str, education: str = "", projects: str = "") -> str:
    """
    Combines all enhanced resume sections into one clean final string.
    Adds section headers and spacing.
    """
    lines = []

    if summary.strip():
        lines.append("SUMMARY")
        lines.append("-------")
        lines.append(summary.strip())
        lines.append("")

    if skills.strip():
        lines.append("SKILLS")
        lines.append("------")
        lines.append(skills.strip())
        lines.append("")

    if experience:
        lines.append("EXPERIENCE")
        lines.append("----------")
        if isinstance(experience, list):
            lines.extend([line.strip() for line in experience])
        else:
            lines.append(experience.strip())
        lines.append("")

    if education.strip():
        lines.append("EDUCATION")
        lines.append("---------")
        lines.append(education.strip())
        lines.append("")

    if projects.strip() and not re.search(r"no (projects|relevant projects|projects listed)", projects.strip(), re.I):
        lines.append("PROJECTS")
        lines.append("--------")
        lines.append(projects.strip())
        lines.append("")

    return "\n".join(lines).strip()

