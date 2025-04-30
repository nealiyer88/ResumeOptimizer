import os
from dotenv import load_dotenv
from gpt_parser import parse_resume_with_gpt
from keyword_matcher import extract_keywords, filter_relevant_keywords, compute_keyword_match
from keyword_classifier import classify_keywords
from keyword_scorer import score_keywords
from llm_enhancer import (
    enhance_summary_with_gpt,
    enhance_skills_with_gpt,
    enhance_experience_job,
    enhance_projects_with_gpt
)
from semantic_filter import filter_keywords_by_semantic_similarity
from resume_formatter import format_experience_section, assemble_resume

load_dotenv()

def run_resume_enhancement_pipeline(resume_text: str, job_posting: str) -> tuple[str, dict]:
    """
    Executes the full resume enhancement pipeline and scoring logic.
    Returns enhanced resume string and a scoring summary dictionary.
    """
        
        # Extract and filter job description keywords
    raw_keywords = extract_keywords(job_posting)
    gpt_filtered_keywords = filter_relevant_keywords(list(raw_keywords))
    semantic_filtered_keywords = filter_keywords_by_semantic_similarity(resume_text, gpt_filtered_keywords)
    classified_keywords = classify_keywords(semantic_filtered_keywords)


        # Compute pre-enhancement keyword match and score
    pre_match = compute_keyword_match(resume_text, job_posting)
    pre_scores = score_keywords(classified_keywords, pre_match["matched_keywords"])

    # Step 1: Parse resume sections
    sections = parse_resume_with_gpt(resume_text)

    contact_info = sections.get("contact_info", {})
    # Robust fallback: sanitize to dict even if GPT returns weird types
    if isinstance(contact_info, list):
        # Try to infer from list of strings
        contact_info = {f"item_{i}": str(item) for i, item in enumerate(contact_info)}
    elif isinstance(contact_info, str):
        contact_info = {"raw": contact_info.strip()}
    elif not isinstance(contact_info, dict):
        contact_info = {}
        # === Normalize name key from common fallbacks ===
    if not contact_info.get("name"):
        contact_info["name"] = (
            contact_info.get("full_name") or
            contact_info.get("title") or
            contact_info.get("raw") or
            ""
        )

    summary_text = sections.get("summary", "")
    if isinstance(summary_text, list):
        summary_text = " ".join(str(s) for s in summary_text)
    elif not isinstance(summary_text, str):
        summary_text = str(summary_text)

    skills_text = sections.get("skills", "")
    if isinstance(skills_text, list):
        skills_text = ", ".join(str(s) for s in skills_text)
    elif not isinstance(skills_text, str):
        skills_text = str(skills_text)

    experience_jobs = sections.get("experience", [])
    print("Parsed Experience Jobs:")
    for i, job in enumerate(experience_jobs):
        print(f"{i+1}. {job.get('title', '?')} @ {job.get('company', '?')}")

    education_text = sections.get("education", "Available upon request")
    projects_text = sections.get("projects", "")
    if isinstance(projects_text, list):
        projects_text = "\n".join(str(p) for p in projects_text)
    elif not isinstance(projects_text, str):
        projects_text = str(projects_text)

    try:
        if projects_text.strip() and len(projects_text.strip().split()) > 10:  # 10+ words = meaningful projects section
            enhanced_projects = enhance_projects_with_gpt(projects_text, gpt_filtered_keywords)
        else:
            enhanced_projects = ""  # No fake projects

    except Exception as e:
        print("\n🛑 ERROR: Projects enhancement failed")
        print(e)
        enhanced_projects = projects_text

    # Clean/flatten education if needed
    if isinstance(education_text, list):
        formatted_edu = []
        for edu in education_text:
            if isinstance(edu, dict):
                parts = [edu.get("degree", ""), edu.get("field", ""), edu.get("institution", "")]
                formatted_edu.append(" | ".join(part for part in parts if part))
            elif isinstance(edu, str):
                formatted_edu.append(edu)
        education_text = "\n".join(formatted_edu)



    # Step 4: Enhance each resume section
    try:
        enhanced_summary = enhance_summary_with_gpt(summary_text, semantic_filtered_keywords)
    except Exception as e:
        print("\n🛑 ERROR: Summary enhancement failed")
        print(e)
        raise

    try:
        enhanced_skills = enhance_skills_with_gpt(skills_text, semantic_filtered_keywords)
    except Exception as e:
        print("\n🛑 ERROR: Skills enhancement failed")
        print(e)
        raise

    enhanced_jobs = []
    try:
        for job in experience_jobs:
            original_bullet_count = len(job.get("bullets", []))
            enhanced_job = enhance_experience_job(
                job,
                semantic_filtered_keywords,
                job_posting,
                original_bullet_count
            )
            enhanced_jobs.append(enhanced_job)
    except Exception as e:
        print("\n🛑 ERROR: Experience enhancement failed")
        print(e)
        raise

            # === Build header block from contact_info ===
    header_lines = []

    if contact_info.get("name"):
        header_lines.append(contact_info["name"])

    # Optional second line (title/location/phone/email)
    sub_header = " | ".join([
        contact_info.get("title", ""),
        contact_info.get("location", ""),
        contact_info.get("phone", ""),
        contact_info.get("email", "")
    ]).strip(" |")
    if sub_header:
        header_lines.append(sub_header)

    # Optional third line (links)
    link_line = " | ".join(filter(None, [
        contact_info.get("linkedin", ""),
        contact_info.get("github", ""),
        contact_info.get("website", "")
    ]))
    if link_line:
        header_lines.append(link_line)

    header_block = "\n".join(header_lines).strip()

    # Step 5: Format sections + assemble resume
    formatted_experience = format_experience_section(enhanced_jobs)
    final_resume = assemble_resume(
        summary=enhanced_summary,
        skills=enhanced_skills,
        experience=formatted_experience,
        education=education_text,
        projects=enhanced_projects
    )
    print("[FINAL HTML for export]\n", final_resume)

    # Step 6: Post-enhancement scoring
    post_match = compute_keyword_match(final_resume, job_posting)
    post_scores = score_keywords(classified_keywords, post_match["matched_keywords"])

    # Step 7: Return final resume + score report
    score_report = {
        "before": {
            "match_percent": pre_match["match_percent"],
            "score_by_category": pre_scores
        },
        "after": {
            "match_percent": post_match["match_percent"],
            "score_by_category": post_scores
        },
        "missing_keywords_after": post_match["missing_keywords"]
    }

    return final_resume, score_report, contact_info
