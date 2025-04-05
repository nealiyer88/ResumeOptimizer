import os
from openai import OpenAI
from dotenv import load_dotenv

print("✅ Loaded NEW llm_api with OpenAI SDK v1.x syntax")

def setup_openai():
    load_dotenv()
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def filter_relevant_keywords(resume_keywords, job_keywords, job_description, model="gpt-3.5-turbo"):
    prompt = (
        "You're a resume optimization assistant.\n\n"
        "Below is a list of keywords found in a job description and a separate list found in a candidate's resume. "
        "Your task is to compare them and return **only the job-related, relevant missing keywords** from the job keywords that are not already in the resume keywords.\n\n"
        "Please remove irrelevant, overly generic, personal, or location-based keywords. Focus on technical skills, tools, qualifications, and core competencies that match the job.\n\n"
        f"Resume Keywords:\n{', '.join(resume_keywords)}\n\n"
        f"Job Description Keywords:\n{', '.join(job_keywords)}\n\n"
        f"Job Description Snippet:\n{job_description[:1000]}...\n\n"
        "Return only the relevant missing keywords in this exact Python list format: ['keyword1', 'keyword2']"
    
    )

    try:
        client = setup_openai()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at resume-job keyword matching."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error filtering keywords: {e}")
        return "[]"

# Enhanced Logic for Section Optimization
def enhance_section(section_name, resume_section_text, relevant_keywords, job_posting_text, model="gpt-3.5-turbo"):
    prompt = create_custom_prompt(section_name, resume_section_text, relevant_keywords, job_posting_text)
    response = generate_text_gpt(prompt, model)

    # ✅ Post-process summary to strip any bullets if GPT added them
    if section_name.lower() == "summary":
        response = "\n".join(
            line for line in response.splitlines()
            if not line.strip().startswith(("-", "•", "*"))
        )
    
    return response.strip()



def create_custom_prompt(section_name, resume_section, relevant_keywords, job_posting_text):
    prompts_by_section = {
        "Professional Summary": (
            "Rewrite this summary as a 2 to 3 sentence elevator pitch. Keep it short, strong, and clearly highlight your most impressive accomplishments and skills. "
            "Do NOT use bullet points. Integrate the following keywords naturally, only where appropriate:"
        ),
        "Work Experience": (
            "Rewrite these Work Experience bullets by improving clarity, cutting fluff, and emphasizing measurable outcomes. "
            "Preserve bullet format. Integrate the following relevant keywords naturally where applicable:"
        ),
        "Skills": (
            "Rewrite this Skills section in a clear, concise format. Avoid full sentences or bullets unless already present. "
            "Categorize and integrate the following relevant keywords naturally:"
        ),
        "Education": (
            "Rewrite this Education section to clearly highlight relevant credentials and institutions. "
            "Preserve original formatting. Integrate the following relevant keywords only where meaningful:"
        ),
        "Projects": (
            "Rewrite this Projects section to emphasize results, relevant skills, and clearly communicate project value. "
            "Preserve any formatting already used. Integrate the following keywords naturally:"
        ),
    }


    base_prompt = prompts_by_section.get(section_name, f"Enhance the '{section_name}' section of the resume by integrating these keywords naturally and clearly emphasizing professional impact:")

    prompt = (
        f"{base_prompt}\n\n"
        f"Resume Section:\n{resume_section}\n\n"
        f"Relevant Keywords: {', '.join(relevant_keywords)}\n\n"
        f"Job Posting Snippet:\n{job_posting_text[:1000]}...\n\n"
        "Return only the enhanced resume bullet points."
    )

    return prompt

def generate_text_gpt(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=None):
    try:
        if max_tokens is None:
            max_tokens = 300 if "summary" in prompt.lower() else 500

        client = setup_openai()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful and professional resume assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating text with GPT: {e}")
        return ""

