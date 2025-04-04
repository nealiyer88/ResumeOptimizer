import os
import ast
from bs4 import BeautifulSoup
import requests

from parsing_module import (
    extract_text_pdfminer,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_keywords,
    calculate_keyword_match,
    split_resume_into_sections,
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


def run_enhancement_pipeline(resume_file, job_input):
    resume_text = process_resume(resume_file)
    sections = split_resume_into_sections(resume_text, pdf_path=resume_file)

    resume_keywords = extract_keywords(resume_text)
    job_text = process_job_posting(job_input)
    job_keywords = extract_keywords(job_text)

    match_percentage = calculate_keyword_match(resume_keywords, job_keywords)
    print(f"[🔍] Resume–Job Keyword Match: {match_percentage:.2f}%")
    enhanced_sections = {}

    # Try to enhance 'experience' using the special enhancer
    if "experience" in sections:
        try:
            print("[⚙️] Enhancing 'experience' section using LLM enhancer...")
            enhanced_experience = enhance_resume_experience(sections["experience"], list(job_keywords))
            enhanced_sections["experience"] = enhanced_experience
            print("[✅] Experience section enhanced with LLM Enhancer.\n")
        except Exception as e:
            print(f"[⚠️] LLM experience enhancement failed. Falling back to default logic.\n{e}")

    # Enhance the rest of the sections
    for section_name, section_text in sections.items():
        if section_name == "experience" and "experience" in enhanced_sections:
            continue

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

  
    print("=== Sections extracted ===")
    print(sections.keys())

    print("=== Sections enhanced ===")
    print(enhanced_sections.keys())

    # === Debug: Check keyword coverage in enhanced sections ===
    print("\n=== Keyword Coverage Debug ===")
    for section, content in enhanced_sections.items():
        matched_keywords = [kw for kw in job_keywords if kw.lower() in content.lower()]
        percent_used = round((len(matched_keywords) / len(job_keywords)) * 100, 2) if job_keywords else 0
        print(f"{section.capitalize()}: {len(matched_keywords)} / {len(job_keywords)} keywords used ({percent_used}%)")
        print(f"Used: {matched_keywords}\n")

    return enhanced_sections
