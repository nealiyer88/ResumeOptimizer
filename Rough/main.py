# main.py (Updated fully integrated version)

import os
import requests
import ast
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from parsing_module import (
    extract_text_pdfminer,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_keywords,
    calculate_keyword_match,
    split_resume_into_sections,
    normalize_section_name
)

from llm_utils import enhance_section, filter_relevant_keywords
from llm_enhancer import enhance_resume_experience



def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style"]):
            element.decompose()
        text = soup.get_text(separator='\n')
        lines = (line.strip() for line in text.splitlines())
        cleaned_text = '\n'.join(line for line in lines if line)
        return cleaned_text
    except Exception as e:
        print(f"Error reading job posting from URL: {e}")
        return ""


def process_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_pdfminer(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format for resume.")


def process_job_posting(input_text_or_url):
    if input_text_or_url.startswith("http"):
        return extract_text_from_url(input_text_or_url)
    else:
        return input_text_or_url


def setup_environment():
    load_dotenv()
    #print("✅ Environment variables loaded successfully.")


def main():
    setup_environment()

    # === 1. Process resume file ===
    resume_file = "docs/sample_resume.pdf"
    resume_text = process_resume(resume_file)
    sections = split_resume_into_sections(resume_text, pdf_path=resume_file)


    # Aggregate keywords from entire resume
    resume_keywords = extract_keywords(resume_text)

    # === 2. Process job posting ===
    job_input = """Reporting to the Head of Corporate Finance, the Analyst will have a key role in Lightshift’s short-term and long-term financial planning. The ideal candidate has experience working in an entrepreneurial environment and with complex corporate financial models. 



Core responsibilities will include:

Develop, generate, and enhance end-to-end financial and operational models;
Manage analysis of monthly, quarterly, and annual reviews;
Improve and implement financial planning and budgeting process;
Assist in the preparation of monthly, quarterly, and annual presentations to the Board and management;
Conduct fundamental analysis of company and portfolio-level cash flow, including in-depth financial analysis to drive decision making;
Coordinate with accounting and treasury to facilitate accurate and timely reporting;
Support finance, accounting, and treasury functions as needed; 


Qualifications:

BA/BS in finance, accounting, economics, or related degree; CPA/CFA a plus;
2+ years of experience with corporate FP&A; preferably in renewable energy;
Proficient in Excel; ability to build, manipulate and interpret complex financials models and perform sensitivities;
Proficient in PowerPoint and other reporting tools; skilled in creating presentations; 
Entrepreneurial team player with ability to drive business execution and performance 
Excellent interpersonal, organizational, and communication skills;
Excellent quantitative and qualitative analytical skills;
Understanding of GAAP financial accounting;
Task-oriented with high attention to detail;
Willingness to travel up to 10% of the time. 
"""
    job_text = process_job_posting(job_input)
    job_keywords = extract_keywords(job_text)

    # === 3. Calculate overlap ===
    match_percentage = calculate_keyword_match(resume_keywords, job_keywords)
    #print("Keyword Match Percentage:", round(match_percentage, 2))

    enhanced_sections = {}

    if match_percentage < 80:
        missing_keywords = job_keywords - resume_keywords
        #print("Raw Missing Keywords:", missing_keywords)

    # ✅ Try to enhance 'experience' using the special enhancer
    if "experience" in sections:
        try:
            print("[⚙️] Enhancing 'experience' section using LLM enhancer...")
            enhanced_experience = enhance_resume_experience(resume_file, list(job_keywords))
            enhanced_sections["experience"] = enhanced_experience
            print("[✅] Experience section enhanced with LLM Enhancer.\n")
        except Exception as e:
            print(f"[⚠️] LLM experience enhancement failed. Falling back to default logic.\n{e}")

    # ✅ Loop through and enhance the rest of the sections (or fallback for experience if not already done)
    for section_name, section_text in sections.items():
        if section_name == "experience" and "experience" in enhanced_sections:
            continue  # already handled by LLM enhancer

        prompt_keywords_str = filter_relevant_keywords(
            extract_keywords(section_text),
            job_keywords,
            job_text
        )
        try:
            section_keywords = ast.literal_eval(prompt_keywords_str)
        except:
            section_keywords = []

        enhanced_text = enhance_section(
            section_name,
            section_text,
            section_keywords,
            job_text
        )

        enhanced_sections[section_name] = enhanced_text


        # === 5. Output enhanced sections ===
        print("\n--- Enhanced Resume Sections ---\n")
        for section, enhanced_content in enhanced_sections.items():
            print(f"## {section.capitalize()} ##\n")
            print(enhanced_content)
            print("\n")


    else:
        print("Resume already meets/exceeds the keyword threshold.")


if __name__ == "__main__":
    main()
