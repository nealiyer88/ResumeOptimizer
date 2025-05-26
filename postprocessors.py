import re
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
from tiktoken import get_encoding
from gpt_utils import gpt_safe_call, num_tokens


client = OpenAI()

EMBED_LIMIT = 8100  # conservative buffer under 8192 max

def embed_text(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    try:
        if num_tokens(text) > EMBED_LIMIT:
            print(f"✂️ Truncating text from {num_tokens(text)} tokens to fit embedding model limit...")
            words = text.split()
            truncated = []
            for word in words:
                truncated.append(word)
                if num_tokens(" ".join(truncated)) >= EMBED_LIMIT:
                    break
            text = " ".join(truncated)

        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding

    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []


REFERENCE_BULLETS_PATH = Path("reference_bullets.json")

def load_reference_bullets(path: Path = REFERENCE_BULLETS_PATH) -> list[str]:
    """
    Loads high-quality reference bullets from external JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            bullets = json.load(f)
            return [b.strip() for b in bullets if b.strip()]
    except Exception as e:
        print(f"[ERROR] Failed to load reference bullets: {e}")
        return []
    
REFERENCE_BULLETS = load_reference_bullets()
REFERENCE_EMBEDDINGS = [embed_text(bullet) for bullet in REFERENCE_BULLETS]


def regenerate_summary_line(original_sentence: str, resume_text: str) -> str:
    """
    Use GPT to rewrite a hallucinated summary sentence using only information from the resume.
    """

    def full_prompt():
        return f"""
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
    \"\"\"{resume_text.strip()}\"\"\"

    Original sentence (possibly hallucinated):
    {original_sentence.strip()}

    ---

    Return only the rewritten version. No quotes. No labels. No formatting.
    """.strip()

    def trimmed_prompt():
        return f"""
    Rewrite the following resume summary sentence using grounded, resume-backed language.

    Sentence:
    "{original_sentence.strip()}"

    Resume snippet:
    {resume_text.strip()[:800]}

    Do not fabricate tools, roles, or experience. Return only the rewritten sentence.
    """.strip()

    def run(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    return gpt_safe_call(
        prompt_fn=full_prompt,
        fallback_prompt_fn=trimmed_prompt,
        run_fn=run,
        fallback_return=original_sentence
    )


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

    def full_prompt():
        return f"""
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
    \"\"\"{resume_text.strip()}\"\"\"

    Relevant keywords from the job posting:
    {", ".join(filtered_keywords)}

    Return only the formatted Skills section with no commentary.
    """.strip()

    def trimmed_prompt():
        return f"""
    Rebuild this resume's Skills section using grouped formatting only for keywords clearly supported below.

    Relevant keywords:
    {", ".join(filtered_keywords[:5])}

    Resume snippet:
    {resume_text.strip()[:1000]}

    Avoid fabrication. Return only the formatted Skills section.
    """.strip()

    def run(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    return gpt_safe_call(
        prompt_fn=full_prompt,
        fallback_prompt_fn=trimmed_prompt,
        run_fn=run,
        fallback_return=""
    )


def evaluate_bullet_quality(bullet: str, threshold: float = 0.76) -> dict:
    """
    Compares a bullet to reference bullets using embedding similarity.
    Returns a quality score and tags like 'low_quality' if it falls below the threshold.
    """
    bullet = bullet.strip()
    if not bullet:
        return {"score": 0.0, "tags": ["empty"]}

    try:
        bullet_vec = embed_text(bullet)
        sims = [cosine_similarity([bullet_vec], [ref_vec])[0][0] for ref_vec in REFERENCE_EMBEDDINGS]
        max_sim = max(sims)

        tags = []
        if max_sim < threshold:
            tags.append("low_quality")

        return {"score": round(max_sim, 3), "tags": tags}

    except Exception as e:
        print(f"[Embedding Error] Could not score bullet: '{bullet}'\n{e}")
        return {"score": 0.0, "tags": ["embedding_error"]}


def rewrite_bullet_with_gpt(original_bullet: str, resume_text: str = "", score: float = None) -> str:
    """
    Uses GPT to rewrite a weak resume bullet into a clearer, stronger format.
    Optionally uses the full resume as context.
    """
    if score is not None:
        print(f"🛠 Rewriting bullet due to low score ({score:.2f}): {original_bullet.strip()}")

    context = f"\nResume context for reference:\n{resume_text.strip()}" if resume_text else ""

    def full_prompt():
        return f"""
    You are rewriting a resume bullet point to be clearer, more impactful, and ATS-optimized.

    🎯 GOAL:
    - Improve clarity, tone, and structure
    - Begin with a strong action verb
    - Focus on a concrete action and measurable result if possible
    - Maintain professional tone aligned with business or technical roles
    - Optimize for Applicant Tracking Systems (ATS)

    🚫 STRICT RULES:
    - NEVER fabricate results, metrics, tools, or achievements not clearly implied
    - Do NOT add soft skills, personality traits, or generic filler (e.g., communication, hardworking)
    - DO NOT echo the original passively — rewrite with impact
    - Keep to a **single concise bullet** (1 sentence max)
    - NO emojis, markdown, formatting, or explanations

    ---

    Original Bullet:
    "{original_bullet.strip()}"

    {context}

    Return ONLY the rewritten bullet. No preamble, no formatting, no explanation.
    """.strip()

    def trimmed_prompt():
        return f"""
    Rewrite this resume bullet clearly and concisely using the snippet below as resume context.

    Bullet:
    "{original_bullet.strip()}"

    Context:
    {resume_text.strip()[:1000]}

    Avoid vague phrasing or fabrication. Return just the improved bullet.
    """.strip()

    def run(prompt):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    return gpt_safe_call(
        prompt_fn=full_prompt,
        fallback_prompt_fn=trimmed_prompt,
        run_fn=run,
        fallback_return=original_bullet
    )



def filter_and_rewrite_bullets(
    bullets: list[str],
    threshold: float = 0.76,
    min_required: int = 3,
    max_total: int = 6,
    resume_text: str = ""
) -> tuple[list[str], bool]:
    """
    Filters strong bullets, rewrites weak ones if too few pass, and caps output.
    Returns final list and a flag if fallback was triggered.
    """

    if not bullets:
        return [], False

    # Step 1: Score all bullets
    scored = [(b, evaluate_bullet_quality(b, threshold)) for b in bullets]
    print("\n📊 Bullet Similarity Scores:")
    for b, meta in scored:
        score = meta["score"]
        tags = ", ".join(meta["tags"]) if meta["tags"] else "✅ passed"
        flag = "✅" if score >= threshold else "❌"
        print(f"{flag} {score:.3f} | {tags:30} | {b.strip()[:90]}")



    # Step 2: Extract high-quality bullets only
    good = [b for b, meta in scored if "low_quality" not in meta["tags"]]

    # === ✅ REWRITE ONLY IF ≤ 1 good bullets ===
    rewritten = []
    fallback_triggered = False
    if len(good) <= 2:
        for b, meta in scored:
            if "low_quality" in meta["tags"]:
                rewritten_bullet = rewrite_bullet_with_gpt(b, resume_text=resume_text, score=meta["score"])
                rewritten.append(rewritten_bullet)

    # Step 3: Combine (original good + rewritten if applicable)
    combined = good + rewritten

    # Step 4: Fallback if we STILL don’t have enough
    if len(combined) < min_required:
        fallback_triggered = True
        sorted_fallback = sorted(scored, key=lambda x: x[1]["score"], reverse=True)
        combined = [b for b, _ in sorted_fallback[:min_required]]

    # Step 5: Enforce hard cap
    final = combined[:max_total]

    return final, fallback_triggered

