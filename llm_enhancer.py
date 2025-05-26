# llm_enhancer.py

import os
from openai import OpenAI
import openai
from dotenv import load_dotenv
from typing import List
from keyword_matcher import extract_keywords, filter_relevant_keywords
import time
from gpt_utils import gpt_safe_call, num_tokens

# Load your OpenAI key securely
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def enhance_summary_with_gpt(summary_text: str, filtered_keywords: list) -> str:
    def full_prompt():
        return f"""
    You are enhancing the *Professional Summary* section of a resume.

    🎯 Your goal:
    - Keep it **factual**, **concise**, and **punchy** — like a personal elevator pitch under 10 seconds
    - Do **not fabricate** tools, job history, industries, or domains not in the resume
    - ❗ If a keyword is **not clearly supported by the resume**, **omit it** — do not guess or generalize
    - Avoid inferred industries/domains (e.g., "banking", "biotech") unless explicitly stated
    - Focus on relevant *tools, certifications, or domain expertise* only if appropriate
    - Avoid redundancy, vague phrases, or filler language
    - Do NOT introduce job titles not already present in the resume
    - Avoid repeating sentence structures
    - Cap summary to exactly **2 sentences and 40 words maximum**

    ---

    ✍️ Current Summary (may be blank):
    {summary_text.strip()}

    ---

    🧠 Filtered keywords to integrate (if appropriate):
    {", ".join(filtered_keywords)}

    ---

    Return ONLY the improved summary. No "Summary:" label. No markdown. No extra formatting.
    """.strip()

    def trimmed_prompt():
        short = summary_text.strip()
        if "." in short:
            short = ". ".join(short.split(".")[:2]) + "."
        short = short[:300]

        return f"""
    Rewrite this into a 2-sentence, 40-word max professional summary.
    Avoid job titles, soft skills, and tools not found in the resume.

    Summary:
    {short}
    """.strip()

    def run(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    return gpt_safe_call(
        prompt_fn=full_prompt,
        fallback_prompt_fn=trimmed_prompt,
        run_fn=run,
        fallback_return=summary_text
    )



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

# Enhance skills section using GPT with fallback for token limit breaches
def enhance_skills_with_gpt(skills_text: str, filtered_keywords: list) -> str:
    format_type = detect_skills_format(skills_text)

    def full_prompt():
        return build_skills_prompt(skills_text, filtered_keywords, format_type)

    def trimmed_prompt():
        # Fallback: trim to first 300 chars and top 5 keywords
        short_skills = str(skills_text).strip()[:300]
        top_keywords = filtered_keywords[:5]
        return build_skills_prompt(short_skills, top_keywords, format_type)

    def run(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    return gpt_safe_call(
        prompt_fn=full_prompt,
        fallback_prompt_fn=trimmed_prompt,
        run_fn=run,
        fallback_return=skills_text  # fallback to original if both prompts fail
    )

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
    original_bullet_count: int
) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    bullets = job.get("bullets", [])
    if isinstance(bullets, str):
        bullets = bullets.splitlines()

    def full_prompt():
        return build_experience_prompt(
            bullets=bullets,
            filtered_keywords=filtered_keywords,
            job_posting=job_posting
        )

    def trimmed_prompt():
        short_bullets = bullets[:3]  # Keep only top 3 bullets
        short_keywords = filtered_keywords[:5]
        short_jd = job_posting[:1000]
        return build_experience_prompt(
            bullets=short_bullets,
            filtered_keywords=short_keywords,
            job_posting=short_jd
        )

    def run(prompt):
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

        # Enforce bullet cap: 6 or original + 2, whichever is smaller
        max_bullets = max(min(original_bullet_count + 2, 6), original_bullet_count)
        return enhanced_bullets[:max_bullets]

    try:
        enhanced_bullets = gpt_safe_call(
            prompt_fn=full_prompt,
            fallback_prompt_fn=trimmed_prompt,
            run_fn=run,
            fallback_return=bullets
        )

        return {
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "date_range": job.get("date_range", ""),
            "bullets": enhanced_bullets,
        }

    except Exception as e:
        print(f"[Experience Enhancement Error] {e}")
        return job  # fallback to original if anything fails


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

def enhance_projects_with_gpt(projects_text: str, filtered_keywords: list) -> str:
    def full_prompt():
        return build_projects_prompt(projects_text, filtered_keywords)

    def trimmed_prompt():
        short_projects = str(projects_text).strip()[:600]  # fallback to ~600 chars
        top_keywords = filtered_keywords[:5]
        return build_projects_prompt(short_projects, top_keywords)

    def run(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    return gpt_safe_call(
        prompt_fn=full_prompt,
        fallback_prompt_fn=trimmed_prompt,
        run_fn=run,
        fallback_return=projects_text
    )

