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
    - Keep it **factual**, **concise**, and **punchy** — 
    - Make it **impactful and succinct** — similar to a concise personal elevator pitch that can be read in under 10 seconds
    - Do **not fabricate** degrees, tools, job history, industries, or domains that aren't clearly present in the resume
    - ! If a keyword is **not clearly supported by the resume**, **OMIT it** — **do not guess or generalize**
    - Do **not infer industries or domains** (e.g., "banking", "biotech") unless mentioned or strongly implied
    - Avoid phrasing that implies lack of expertise (e.g., "entry-level", "junior", "beginner")
    - Focus only on integrating missing *tools, certifications, or domain expertise* if appropriate
    - Keep tone professional, clear, and confident
    - **Avoid redundancy and vague phrasing**
    - DO NOT fabricate achievements or job titles
    -❗Do not introduce or modify job titles. Use only titles explicitly found in the resume (e.g., Analyst, Manager, Senior Manager).
    - Avoid repeating Sentence starters more than once
    - Sentence 2 should expand on sentence 1 — not duplicate it
    - Always end summary with single punctuation. No trailing double periods.

    ---

    ✍️ Current Summary (may be blank):
    {summary_text.strip()}

    ---

    🧠 Filtered keywords to integrate (if appropriate):
    {", ".join(filtered_keywords)}

    ---

    Return ONLY the improved summary.

    It is a hard cap: no more than 2 sentences AND no more than 40 words, total. Do not exceed either.

    No bullet points. No "Summary:" label. No quotes. No formatting. Just plain text.

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

    🧠 Filtered keywords from the job description:
    {", ".join(filtered_keywords)}

    🎯 Your task:
    - Naturally incorporate as many relevant filtered keywords as possible.
    - Only include hard skills: tools, platforms, certifications, or directly measurable capabilities.
    - ❗ Omit any keyword not clearly supported by the resume — do NOT guess, infer, or generalize.
    - ❗ Do not infer tools or platforms based on job titles or job descriptions alone.
    - If soft skills are present as a category, only enhance that section with relevant soft skills.
    - Do not fabricate new skills unless they are clearly implied by the original resume.
    - Do not repeat or re-list skills already present.
    - If skills are grouped by theme or category, align group headers with the job description terminology.
    - {format_instruction}

    Return ONLY the enhanced skills section. No explanations, no headers.
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
    You are rewriting the bullet points of a single job from a resume. The original bullets may be vague, redundant, or lack measurable impact.

    🎯 Objective:
    Rewrite the bullets to be clear, concise, and aligned with professional job descriptions. Use strong verbs, measurable outcomes, and impactful phrasing where possible.

    📌 Original Bullets:
    {bullets_text}

    🧠 Filtered Keywords (for guidance):
    {keyword_str}

    📄 Job Description (for tone and alignment):
    \"\"\"{job_posting.strip()}\"\"\"

    🔒 Rules:
    - Only use keywords that are clearly present or implied in the original bullets.
    - Do NOT invent metrics, achievements, or tools that aren’t mentioned.
    - Avoid fluff like "responsible for", "worked on", or "helped with".
    - Each bullet must stand alone — no continuation lines or soft phrasing.
    - Use concise, action-driven language.
    - Limit each bullet to one impactful idea.

    Return ONLY the rewritten bullets. One per line. No extra commentary or formatting.
    """.strip()



    return prompt

# Takes a single job dict (title, company, date_range, bullets)
# Builds a GPT prompt with build_experience_prompt(...)
# Calls GPT-4 to rewrite the bullets
# Returns a new job dict with the same title/company/date, but enhanced bullets
def enhance_experience_job(
    job: dict,
    filtered_keywords: List[str],
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
        filtered_keywords=filtered_keywords,
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
        # Enforce bullet cap: 6 for all jobs, unless original had fewer
        max_bullets = max(min(original_bullet_count + 2, 6), original_bullet_count)
        enhanced_bullets = enhanced_bullets[:max_bullets]


        return {
            "title": job["title"],
            "company": job["company"],
            "date_range": job["date_range"],
            "bullets": enhanced_bullets,
        }

    except Exception as e:
        print(f"[Experience Enhancement Error] {e}")
        return job  # fallback to original

def build_projects_prompt(projects_text, filtered_keywords: list) -> str:
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
- Integrate these filtered keywords naturally if relevant: {", ".join(filtered_keywords)}
- DO NOT invent new projects or accomplishments.
- Total project section length must be the same or **shorter** than the original.
- Format bullets clearly and consistently. No empty lines or full paragraphs.

Return only the improved Projects section — no section header, no explanations.
""".strip()

def enhance_projects_with_gpt(projects_text, filtered_keywords: list) -> str:
    prompt = build_projects_prompt(projects_text, filtered_keywords)
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
