import os
from dotenv import load_dotenv
from openai import OpenAI

from experience_splitter import split_experience_section, parse_job_entry

# === Set up OpenAI Client ===
def setup_openai():
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Enhance Single Job Entry ===
def enhance_job_entry_with_gpt(job, job_keywords, client, model="gpt-4"):
    context = (
        f"Company: {job['company']}\n"
        f"Title: {job['title']}\n"
        f"Dates: {job['date_range']}\n\n"
        f"Responsibilities:\n" + "\n".join(job['bullets'])
    )

    prompt = (
        "You are a professional resume writer.\n\n"
        "Improve the job description below by:\n"
        "- Enhancing clarity, conciseness, and tone\n"
        "- Limiting to 3–5 strong, high-impact bullets per job\n"
        "- Avoiding repetitive phrasing (e.g., 'leveraged', 'developed')\n"
        "- Preserving bullet formatting\n"
        f"- Integrating job keywords (only where relevant): {', '.join(job_keywords)}\n"
        "- Ensuring each bullet clearly answers 'So what?' with impact or business value\n"
        "- Quantify results (e.g., time saved, accuracy improved)\n"
        "- Do not exaggerate or invent accomplishments\n\n"
        "Rewrite this job experience:\n"
        f"```\n{context}\n```\n\n"
        "Respond only with improved bullet points in bullet format."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

# === Format Final Output ===
def format_enhanced_experience(jobs):
    blocks = []
    seen_blocks = set()

    for job in jobs:
        block = (
            f"Company: {job['company']}\n"
            f"Title: {job['title']}\n"
            f"Dates: {job['date_range']}\n\n"
            f"{job['bullets']}"
        ).strip()

        if block not in seen_blocks:
            blocks.append(block)
            seen_blocks.add(block)

    return "\n\n".join(blocks)

# === Main Experience Enhancer ===
def enhance_resume_experience(experience_text, job_keywords, model="gpt-4"):
    if not experience_text:
        raise ValueError("No experience section text provided.")

    chunks = split_experience_section(experience_text)
    parsed_jobs = [parse_job_entry(chunk) for chunk in chunks]
    parsed_jobs = [job for job in parsed_jobs if job['company'] and job['title']]

    # ✅ Remove duplicate job entries based on (company, title, date_range)
    seen = set()
    deduped_jobs = []
    for job in parsed_jobs:
        job_id = (job["company"], job["title"], job["date_range"])
        if job_id not in seen:
            seen.add(job_id)
            deduped_jobs.append(job)
    parsed_jobs = deduped_jobs

    client = setup_openai()
    enhanced_jobs = []

    for job in parsed_jobs:
        bullets = enhance_job_entry_with_gpt(job, job_keywords, client, model)
        enhanced_jobs.append({
            "company": job['company'],
            "title": job['title'],
            "date_range": job['date_range'],
            "bullets": bullets
        })

    return format_enhanced_experience(enhanced_jobs)

# === Entry Point for Manual Testing ===
if __name__ == "__main__":
    from parsing_module import extract_text_pdfminer, split_resume_into_sections
    pdf_path = "docs/sample_resume.pdf"
    resume_text = extract_text_pdfminer(pdf_path)
    sections = split_resume_into_sections(resume_text, pdf_path)
    experience = sections.get("experience", "")
    job_keywords = ["SQL", "Python", "Power BI", "budget forecasting", "reporting automation", "workforce analytics"]
    output = enhance_resume_experience(experience, job_keywords)
    print("\n=== FINAL ENHANCED EXPERIENCE SECTION ===\n")
    print(output)
