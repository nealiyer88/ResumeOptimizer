# llm_enhancer.py

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
from keyword_matcher import extract_keywords, filter_relevant_keywords


# Load your OpenAI key securely
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def enhance_summary_with_gpt(summary_text: str, filtered_keywords: list) -> str:
    """
    Enhance the 'summary' section of a resume using GPT-4,
    by naturally incorporating filtered job description keywords.
    """
    prompt = f"""
You are enhancing the *Professional Summary* section of a resume.

🎯 Your goal:
- Keep it **factual**, **concise**, and **punchy** — ***MAX 2 Sentences***
- Make it **impactful and succinct** — similar to a concise personal elevator pitch that can be read in under 10 seconds
- Do **not fabricate** degrees, tools, job history, industries, or domains that aren't clearly present in the resume
- ! If a keyword is **not clearly supported by the resume**, **OMIT it** — **do not guess or generalize**
- Avoid phrasing that implies lack of expertise (e.g., "entry-level", "junior", "beginner")
- Focus only on integrating missing *tools, certifications, or domain expertise* if appropriate
- Keep tone professional, clear, and confident
- **Avoid redundancy**
- DO NOT fabricate achievements or job titles

---

✍️ Current Summary (may be blank):
{summary_text.strip()}

---

🧠 Filtered keywords to integrate (if appropriate):
{", ".join(filtered_keywords)}

---

Return ONLY the improved summary. No bullet points. No “Summary:” label. No extra formatting.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Summary Enhancement Error] {e}")
        return summary_text  # fallback to original


#Detect format of skills section (comma vs bullet separated)
def detect_skills_format(skills_text: str | list | dict) -> str:
    if isinstance(skills_text, dict):
        # Flatten values into a comma-separated string
        skills_text = ", ".join(
            item for sublist in skills_text.values() for item in (sublist if isinstance(sublist, list) else [sublist])
        )
    """
    Detects whether the skills section uses bullets or commas.

    Returns:
        'bullet' or 'comma'
    """
    if isinstance(skills_text, list):
        return "comma"  # Treat lists as comma-separated

    if any(skills_text.strip().startswith(prefix) for prefix in ("•", "-", "*")):
        return "bullet"
    elif "," in skills_text:
        return "comma"
    else:
        return "unknown"

def build_skills_prompt(skills_text, filtered_keywords: list, format_type: str) -> str:
    if isinstance(skills_text, dict):
        skills_text = skills_text.get("text", "")  # fallback if some bug passed it in wrapped
    elif not isinstance(skills_text, str):
        skills_text = str(skills_text)

    """
    Builds a GPT prompt for enhancing the skills section.
    """
    format_instruction = {
    "comma": "Group skills by theme using headers like 'Machine Learning & NLP', 'Cloud & Data', etc. Return each group on a new line. Separate individual skills with commas.",
    "bullet": "Group skills by theme using headers like 'Machine Learning & NLP', 'Cloud & Data', etc. Return each group as a bulleted list under that header.",
    "unknown": "Organize skills by logical groupings with readable headers and line breaks. Avoid a single block of text."
    }[format_type]


    prompt = f"""
    You are enhancing the 'Skills' section of a resume.

    Here is the original Skills section:

    ---
    {skills_text.strip()}
    ---

    Below is a list of filtered keywords from the job description:
    {", ".join(filtered_keywords)}

    🎯 Your task:
    - Naturally incorporate as many relevant filtered keywords as possible.
    - Only include hard skills (tools, platforms, certifications, or directly measurable capabilities).
    - ❗️ If a keyword is not clearly supported by the resume, **OMIT it**. Do not guess or assume expertise.
    - If the existing section includes soft skills as a subcategory, only add soft skills to that portion.
    - Avoid fabricating new skills unless they are *clearly implied* by the original resume content.
    - DO NOT repeat or re-list skills already present.
    - If the skills section is organized by category, rename the category headers to align with the key themes and terminology of the job description.
    - {format_instruction}

    Return only the final enhanced skills section. No explanations.
    """.strip()

    return prompt

def enhance_skills_with_gpt(skills_text: str, filtered_keywords: list) -> str:
    format_type = detect_skills_format(skills_text)
    prompt = build_skills_prompt(skills_text, filtered_keywords, format_type)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"[Skills Enhancement Error] {e}")
        return skills_text  # fallback to original

# Assembles the instruction to GPT
# Embeds the original bullets
# Injects the missing keywords
# Includes the job description for tone
def build_experience_prompt(bullets: List[str], filtered_keywords: List[str], job_posting: str) -> str:
    """
    Build a GPT prompt to enhance a single job's bullet points.
    - Incorporates relevant keywords
    - Aligns tone and focus with the target job posting
    """
    bullets_text = "\n".join(f"- {b}" for b in bullets)
    keyword_str = ", ".join(filtered_keywords)

    prompt = f"""
You are enhancing the bullet points of a single job from a professional resume.

📌 Original Bullets:
---
{bullets_text}
---

📋 Relevant but filtered keywords from the job posting:
{keyword_str}

💼 Target Job Posting (for tone and relevance alignment):
\"\"\"{job_posting.strip()}\"\"\"

🎯 Your task:
- Rewrite bullets to improve clarity, strength, and relevance.
- ❗ Only use keywords that are clearly supported by the job content — omit if unsure.
- Emphasize accomplishments that are backed by the resume — do not invent.
- Use 1 bullet per line. Combine ideas when appropriate.
- Avoid filler like “responsible for”, “worked on”, etc.

📌 Bullet Rules:
- If original had <4 bullets, you may add 1-2 short bullets if justified.
- If >6 bullets, trim to the best 4-6 lines.
- If job is early-career, prefer 3-4 bullets max.
- DO NOT fabricate industries, job titles, or results.
- Do not invent metrics. Only include if present in resume or implied.

Return ONLY the enhanced bullets. Keep formatting consistent.
""".strip()

    return prompt

# Takes a single job dict (title, company, date_range, bullets)
# Builds a GPT prompt with build_experience_prompt(...)
# Calls GPT-4 to rewrite the bullets
# Returns a new job dict with the same title/company/date, but enhanced bullets
def enhance_experience_job(
    job: dict,
    missing_keywords: List[str],
    job_posting: str,
    original_bullet_count: int  # 👈 Add this!
) -> dict:

    """
    Enhance a single job entry using GPT-4.
    Inputs:
        job: dict with keys 'title', 'company', 'date_range', 'bullets'
        missing_keywords: relevant keywords not found in original resume
        job_posting: full JD text for tone/keyword guidance
    Returns:
        New job dict with same structure but enhanced bullet points
    """
    from openai import OpenAI  # Ensure OpenAI is loaded after dotenv
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    bullets = job["bullets"]
    if isinstance(bullets, str):
        bullets = bullets.splitlines()

    prompt = build_experience_prompt(
        bullets=bullets,
        missing_keywords=missing_keywords,
        job_posting=job_posting,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        enhanced_text = response.choices[0].message.content.strip()
        enhanced_bullets = [
            line.strip("•- ").strip()
            for line in enhanced_text.splitlines()
            if line.strip()
        ]
                # Enforce bullet limit: Keep at most 6
        # ✅ Enforce bullet limit: max 2 more than original
        if len(enhanced_bullets) > original_bullet_count + 2:
            enhanced_bullets = enhanced_bullets[: original_bullet_count + 2]

        # ✅ Also trim down if it's still longer than 6
        if len(enhanced_bullets) > 6:
            enhanced_bullets = enhanced_bullets[:6]

        return {
            "title": job["title"],
            "company": job["company"],
            "date_range": job["date_range"],
            "bullets": enhanced_bullets,
        }

    except Exception as e:
        print(f"[Experience Enhancement Error] {e}")
        return job  # fallback to original

def build_projects_prompt(projects_text, missing_keywords: list) -> str:
    if isinstance(projects_text, dict):
        projects_text = projects_text.get("text", "")
    elif isinstance(projects_text, list):
        projects_text = "\n".join(str(item) for item in projects_text)
    elif not isinstance(projects_text, str):
        projects_text = str(projects_text)    
        """
    Constructs a GPT prompt to enhance the projects section without increasing length.
    """
    return f"""
You are enhancing the 'Projects' section of a resume.

The original content is below:

---
{projects_text.strip()}
---

🎯 Your task:
- Rewrite each project entry using the following format:
  - Project title must be wrapped in this exact tag: `<p class="project-title">Your Title Here</p>`
  • Bullet 1 (skill/action/result based)
  • Bullet 2 (no more than 2-3 lines per project)
- DO NOT use brackets [ ] around titles
- DO NOT prefix titles with icons or bullets
- DO NOT increase font size of Project Title
- Make titles visually distinct by writing them in all caps
- Combine empty or fragmented lines into concise bullets.
- Eliminate duplicate or redundant phrasing.
- Integrate these missing keywords naturally if relevant: {", ".join(missing_keywords)}
- DO NOT invent new projects or accomplishments.
- Total project section length must be the same or **shorter** than the original.
- Format bullets clearly and consistently. No empty lines or full paragraphs.

Return only the improved Projects section — no section header, no explanations.
""".strip()

def enhance_projects_with_gpt(projects_text, missing_keywords: list) -> str:
    prompt = build_projects_prompt(projects_text, missing_keywords)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Projects Enhancement Error] {e}")
        return projects_text  # fallback
