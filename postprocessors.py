import re

def sanitize_summary(summary_text: str, resume_text: str) -> str:
    """
    Strip hallucinated domains, industries, and titles from summary if unsupported in resume.
    No fallback phrases are inserted — keeps it clean, domain-neutral, and production-safe.
    """
    if not isinstance(summary_text, str):
        summary_text = str(summary_text)
    if not isinstance(resume_text, str):
        resume_text = str(resume_text)

    resume_lower = resume_text.lower()
    summary_cleaned = summary_text

    # Find capitalized or multiword phrases (possible hallucinated domains or titles)
    matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", summary_text)

    for phrase in matches:
        phrase_lower = phrase.lower()

        # Only strip if phrase is not mentioned in resume at all
        if phrase_lower not in resume_lower:
            # Optional: print to logs for monitoring
            print(f"🧼 Removed unsupported phrase: '{phrase}'")
            summary_cleaned = summary_cleaned.replace(phrase, "").strip()

    # Clean up double spaces or leftover formatting issues
    summary_cleaned = re.sub(r"\s{2,}", " ", summary_cleaned)
    summary_cleaned = re.sub(r"\s+\.", ".", summary_cleaned)

    return summary_cleaned

def sanitize_skills(skills_text: str, resume_text: str) -> str:
    """
    Remove hallucinated skills (tools, platforms, domains) from the Skills section
    if they are not clearly present in the resume.
    Works across comma-separated, bulleted, or grouped formats.
    """
    if not isinstance(skills_text, str):
        skills_text = str(skills_text)
    if not isinstance(resume_text, str):
        resume_text = str(resume_text)

    resume_lower = resume_text.lower()
    cleaned_skills = []
    
    # Normalize delimiters
    raw_skills = re.split(r"[•*•\-–,\n]", skills_text)

    for skill in raw_skills:
        skill_clean = skill.strip()
        if not skill_clean:
            continue

        # Basic substring check (exact match or key fragment in resume)
        skill_lower = skill_clean.lower()
        if skill_lower in resume_lower:
            cleaned_skills.append(skill_clean)
        else:
            print(f"🧼 Removed hallucinated skill: '{skill_clean}'")

    # Return comma-separated string
    return ", ".join(cleaned_skills)