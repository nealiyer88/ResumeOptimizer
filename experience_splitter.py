import re
import pdfplumber
from parsing_module import split_resume_into_sections

# === Step 1: Extract raw text from PDF ===
pdf_path = "docs/sample_resume.pdf"

with pdfplumber.open(pdf_path) as pdf:
    resume_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# === Step 2: Try using section parser ===
parsed_sections = split_resume_into_sections(resume_text, pdf_path=pdf_path)
#print("\n=== Top-Level Resume Sections Extracted ===")
for k, v in parsed_sections.items():
    #print(f"\n--- {k.upper()} ---")
    #print(v[:500])  # show first 500 chars to preview content
    experience_text = parsed_sections.get("experience", "")

# === Step 3: Fallback if experience section is empty ===
if not experience_text.strip():
    #print("[Fallback] No 'experience' section found. Trying manual keyword search.")
    lower_resume = resume_text.lower()
    start = lower_resume.find("experience")
    experience_text = resume_text[start:] if start != -1 else ""

    # Optionally cut off at next known section
    stop_keywords = ["education", "projects", "certifications", "skills"]
    for kw in stop_keywords:
        stop = experience_text.lower().find(kw)
        if stop != -1:
            experience_text = experience_text[:stop]
            break

# === Splitter ===
def split_experience_section(text):
    lines = text.strip().split("\n")
    job_chunks = []
    current_chunk = []

    date_pattern = re.compile(
        r"((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+['’]?\d{2,4})|"
        r"(\d{4}\s*[-–—]\s*(Present|\d{2,4}))",
        flags=re.IGNORECASE
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        if date_pattern.search(line):
            if current_chunk:
                job_chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
            if i >= 2:
                current_chunk.extend([lines[i-2], lines[i-1]])
            elif i == 1:
                current_chunk.append(lines[i-1])
        current_chunk.append(line)
        i += 1

    if current_chunk:
        job_chunks.append("\n".join(current_chunk).strip())

    return job_chunks


# === Parser ===
def parse_job_entry(chunk):
    lines = chunk.strip().split("\n")
    date_pattern = re.compile(
        r"((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+[‘’']?\d{2,4})|"
        r"(\d{4}\s*[-–—]\s*(Present|\d{2,4}))",
        flags=re.IGNORECASE
    )

    date_line_idx = None
    date_match = None

    for i, line in enumerate(lines):
        match = date_pattern.search(line)
        if match:
            date_line_idx = i
            date_match = match
            break

    if date_line_idx is None:
        return {
            "company": None,
            "title": None,
            "date_range": None,
            "bullets": lines
        }

    date_line = lines[date_line_idx].strip()
    date_range = date_line[date_match.start():].strip()
    title = date_line[:date_match.start()].strip()
    company = lines[date_line_idx - 1].strip() if date_line_idx >= 1 else None

    # 👇 Bullet symbols expanded here
    bullets = []
    current_bullet = ""

    for line in lines[date_line_idx + 1:]:
        striped = line.strip()
        if striped.startswith(("-", "•", "·")):
            if current_bullet:
                bullets.append(current_bullet.strip())
            current_bullet = striped
        else:
            current_bullet += " " + striped

    if current_bullet:
        bullets.append(current_bullet.strip())


    return {
        "company": company,
        "title": title,
        "date_range": date_range,
        "bullets": bullets
    }

# === Main Execution ===
if __name__ == "__main__":
    chunks = split_experience_section(experience_text)
    # Remove first chunk if it doesn't include any date (likely a leftover header)
    date_check = re.compile(r"\d{4}")
    if chunks and not date_check.search(chunks[0]):
        #print("[DEBUG] Removing first non-job chunk (likely a header)")
        chunks = chunks[1:]


    #print("\n=== Raw Job Chunks ===")
    for i, chunk in enumerate(chunks):
        #print(f"\n--- Job {i+1} ---\n{chunk}\n")

        if chunks:
        #print("\n=== Parsed Job Fields (First Job Only) ===")
            parsed = parse_job_entry(chunks[0])
        #for k, v in parsed.items():
            #if isinstance(v, list):
                #print(f"{k}:\n" + "\n".join(v) + "\n")
            #else:
                #print(f"{k}:\n{v}\n")

    #print("\n=== All Parsed Job Entries ===")
    #parsed_jobs = []
    #for i, chunk in enumerate(chunks):
        #parsed = parse_job_entry(chunk)
        #parsed_jobs.append(parsed)

        #print(f"\n--- Job {i + 1} ---")
        #print(f"Company: {parsed['company']}")
        #print(f"Title: {parsed['title']}")
        #print(f"Dates: {parsed['date_range']}")
        #print("Bullets:")
        #for b in parsed["bullets"]:
            #print(f"- {b}")
