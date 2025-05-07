import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def embed_text(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def regenerate_summary_line(original_sentence: str, resume_text: str) -> str:
    """
    Use GPT to rewrite a hallucinated summary sentence using only information from the resume.
    """
    prompt = f"""
    You are rewriting a single summary sentence from a resume. The original sentence may include hallucinated tools, technologies, or traits. Your job is to rewrite it using only content clearly supported by the resume.

    🎯 Rules:
    - DO NOT include any tools, software, platforms, or technologies — those belong in the Skills section, not Summary
    - DO NOT include soft skill fluff like “communication skills” or “presentation”
    - Focus only on measurable impact, capabilities, or resume-backed accomplishments
    - Maintain a professional, recruiter-appropriate tone
    - Rewrite as a tight summary bullet (1 sentence max if possible)
    - Avoid repeating sentence starters more than once
    - Sentence 2 should expand on sentence 1 — not duplicate it
    - Always end summary with single punctuation. No trailing double periods.

    📏 Constraint:
        It is a hard cap: maximum 40 words. Do not exceed.

    ---

    Resume (source of truth):
    \"\"\"
    {resume_text.strip()}
    \"\"\"

    Original sentence (possibly hallucinated):
    {original_sentence.strip()}

    ---

    Return only the rewritten version.
    No quotes. No labels. No formatting.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def sanitize_summary(summary_text: str, resume_text: str, threshold: float = 0.82) -> str:
    """
    Edit unsupported summary lines instead of deleting them.
    Uses literal check → semantic match → GPT rewriter fallback.
    """
    if not isinstance(summary_text, str):
        summary_text = str(summary_text)
    if not isinstance(resume_text, str):
        resume_text = str(resume_text)

    resume_lower = resume_text.lower()
    resume_embedding = embed_text(resume_text)
    summary_sentences = re.split(r"[.]", summary_text)

    final_summary = []

    for sent in summary_sentences:
        sent = sent.strip()
        if not sent:
            continue

        # ✅ 1. Literal match
        if sent.lower() in resume_lower:
            final_summary.append(sent)
            continue

        # ✅ 2. Semantic similarity match
        sent_embedding = embed_text(sent)
        sim = cosine_similarity([resume_embedding], [sent_embedding])[0][0]
        if sim >= threshold:
            final_summary.append(sent)
        else:
            # ✅ 3. Rewrite unsupported sentence with GPT
            print(f"🧠 Rewriting unsupported summary sentence: '{sent}'")
            rewritten = regenerate_summary_line(sent, resume_text)
            if rewritten:
                final_summary.append(rewritten)

    return ". ".join(final_summary).strip() + "."

def sanitize_skills(skills_text: str, resume_text: str, threshold: float = 0.82) -> str:
    """
    Hybrid cleaner: Remove hallucinated skills using literal check + semantic fallback.
    Optional if rewrite_skills_from_resume() is used.
    """
    if not isinstance(skills_text, str):
        skills_text = str(skills_text)
    if not isinstance(resume_text, str):
        resume_text = str(resume_text)

    resume_lower = resume_text.lower()
    resume_embedding = embed_text(resume_text)
    cleaned_skills = []

    raw_skills = re.split(r"[•*•\-–,\n]", skills_text)

    for skill in raw_skills:
        skill_clean = skill.strip()
        if not skill_clean:
            continue

        skill_lower = skill_clean.lower()

        # First: literal presence
        if skill_lower in resume_lower:
            cleaned_skills.append(skill_clean)
            continue

        # Then: semantic fallback
        skill_embedding = embed_text(skill_clean)
        sim = cosine_similarity([resume_embedding], [skill_embedding])[0][0]
        if sim >= threshold:
            cleaned_skills.append(skill_clean)
        else:
            print(f"🧼 Removed hallucinated skill: '{skill_clean}'")

    return ", ".join(cleaned_skills)

def rewrite_skills_from_resume(resume_text: str, filtered_keywords: list[str]) -> str:
    """
    Rebuild the Skills section using GPT-4 based on resume truth and job relevance.
    Includes grouped formatting if logical. Domain-agnostic.
    """
    prompt = f"""
    You are generating the Skills section of a resume using only information from the resume text.

    💡 Your goal:
    - Group skills by type (e.g., Finance Tools, BI Platforms, Productivity Tools)
    - Each category should be on its own line
    - Within each line, separate skills using the "|" (pipe) character
    - Do NOT return each skill on its own line
    - Do NOT use bullet points, emojis, or markdown formatting
    - Do NOT fabricate any tools, platforms, or skills
    - Use clean headers like 'Finance Tools:' or 'BI Platforms:', not emojis

    Resume (source of truth):
    \"\"\"
    {resume_text.strip()}
    \"\"\"

    Relevant keywords from the job posting:
    {", ".join(filtered_keywords)}

    Return only the formatted Skills section with no commentary.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
