import streamlit as st
import requests
import json
from pathlib import Path
import tempfile
import os
from resume_export import generate_pdf, generate_docx, get_available_styles
import re
import io
from PyPDF2 import PdfReader
from docx import Document

def extract_contact_info(resume_text):
    """Extract basic contact information from resume text."""
    contact_info = {
        'name': '',
        'email': '',
        'phone': '',
        'location': '',
        'linkedin': '',
        'github': '',
        'website': ''
    }
    
    # Get the first few lines where contact info typically appears
    lines = resume_text.split('\n')[:15]  # Look in first 15 lines
    
    # Look for patterns
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    phone_pattern = r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'
    
    # Check for name (usually the first substantial line)
    for line in lines:
        line = line.strip()
        if line and len(line) > 2 and not '@' in line and not line.startswith(('http', 'www')):
            # Skip lines that are likely not names
            if any(term in line.lower() for term in ['resume', 'cv', 'curriculum', 'vitae', 'summary']):
                continue
            contact_info['name'] = line
            break
    
    # Check for email
    for line in lines:
        email_match = re.search(email_pattern, line)
        if email_match:
            contact_info['email'] = email_match.group(0)
            break
    
    # Check for phone
    for line in lines:
        phone_match = re.search(phone_pattern, line)
        if phone_match:
            contact_info['phone'] = phone_match.group(0)
            break
    
    # Check for location (common formats)
    location_keywords = ['city', 'state', 'country', 'location', 'address']
    for line in lines:
        line_lower = line.lower()
        # Skip lines with emails or phones
        if '@' in line or re.search(phone_pattern, line):
            continue
        # Look for location indicators
        if any(keyword in line_lower for keyword in location_keywords) or ',' in line:
            if not contact_info['location']:  # Only set if not already found
                # Clean up the line
                location = re.sub(r'(address|location|city|state|country)[\s:]+', '', line, flags=re.IGNORECASE)
                contact_info['location'] = location.strip()
    
    # Look for GitHub, LinkedIn or website
    for line in lines:
        line_lower = line.lower()
        if 'github' in line_lower:
            github_match = re.search(r'github\.com/[\w-]+', line_lower)
            if github_match:
                contact_info['github'] = github_match.group(0)
        if 'linkedin' in line_lower:
            linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', line_lower)
            if linkedin_match:
                contact_info['linkedin'] = linkedin_match.group(0)
        if any(domain in line_lower for domain in ['.com', '.org', '.net', '.io']) and 'http' in line_lower:
            website_match = re.search(r'https?://[\w.-]+\.\w+[\w/.-]*', line_lower)
            if website_match and not contact_info['website']:
                contact_info['website'] = website_match.group(0)
    
    return contact_info

# Set page config
st.set_page_config(
    page_title="Resume Optimizer",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Resume Optimizer")
st.markdown("""
Enhance your resume to better match job descriptions and improve your chances of getting noticed by recruiters.
""")

# Initialize session state for resume text, job description and contact info
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'enhanced_resume' not in st.session_state:
    st.session_state.enhanced_resume = None
if 'score_report' not in st.session_state:
    st.session_state.score_report = None
if 'contact_info' not in st.session_state:
    st.session_state.contact_info = {
        'name': '',
        'email': '',
        'phone': '',
        'location': '',
        'linkedin': '',
        'github': '',
        'website': ''
    }

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload or Build Your Resume")
    
    # Resume input options
    resume_option = st.radio(
        "Choose how to input your resume:",
        ["Upload PDF", "Paste Text"],
        horizontal=True
    )
    
    if resume_option == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"]
        )
        if uploaded_file is not None:
            file_type = uploaded_file.type

            if uploaded_file.name.endswith(".pdf"):
                # Extract text from PDF
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    st.session_state.resume_text += page.extract_text() or ""
            elif uploaded_file.name.endswith(".docx"):
                # Extract text from DOCX
                doc = Document(uploaded_file)
                for para in doc.paragraphs:
                    st.session_state.resume_text += para.text + "\n"
            elif uploaded_file.name.endswith(".txt"):
                # Extract text from TXT
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                st.session_state.resume_text = stringio.read()
            else:
                st.error("Unsupported file type.")
            
            # Extract contact info from the resume text
            detected_contact_info = extract_contact_info(st.session_state.resume_text)
            
            # Update session state with detected info (only if empty)
            for key, value in detected_contact_info.items():
                if value and not st.session_state.contact_info.get(key):
                    st.session_state.contact_info[key] = value
            
            st.success("Resume uploaded successfully!")
    
    else:  # Paste Text option
        st.session_state.resume_text = st.text_area(
            "Paste your resume text here:",
            height=300,
            value=st.session_state.resume_text
        )

with col2:
    st.subheader("2. Input Job Description")
    st.session_state.job_description = st.text_area(
        "Paste the job description here:",
        height=300,
        value=st.session_state.job_description
    )

# Add a divider
st.divider()

# Process button
if st.button("Optimize Resume", type="primary"):
    if not st.session_state.resume_text:
        st.error("Please provide your resume first.")
    elif not st.session_state.job_description:
        st.error("Please provide a job description.")
    else:
        with st.spinner("Optimizing your resume..."):
            try:
                # Call the backend optimization endpoint
                response = requests.post(
                    'http://localhost:8000/optimize-resume',
                    json={
                        "html_resume": st.session_state.resume_text,
                        'job_posting': st.session_state.job_description
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store the result in session state for later use
                    st.session_state.enhanced_resume = result['enhanced_resume']
                    st.session_state.score_report = result['score_report']

                                        # === NEW: Handle contact_info returned from backend ===
                    if 'contact_info' in result:
                        for key, value in result['contact_info'].items():
                            # Try multiple fallback keys for "name"
                            if value and not st.session_state.contact_info.get(key):
                                st.session_state.contact_info[key] = value
                            # Normalize "name" into Full Name field
                            if 'name' not in st.session_state.contact_info or not st.session_state.contact_info['name']:
                                st.session_state.contact_info['name'] = result['contact_info'].get('full_name') or \
                                                                        result['contact_info'].get('title') or \
                                                                        result['contact_info'].get('raw') or ''



                    
                    # Display enhanced resume
                    st.subheader("Enhanced Resume")
                    st.text_area("Enhanced resume content", st.session_state.enhanced_resume, height=400)
                    
                    # Display score report
                    st.subheader("Score Report")
                    score_report = st.session_state.score_report
                    
                    # Create columns for before/after comparison
                    score_col1, score_col2 = st.columns(2)
                    
                    with score_col1:
                        st.markdown("**Before Optimization**")
                        st.metric("Match Percentage", f"{score_report['before']['match_percent']}%")
                        st.metric("Final Score", f"{score_report['before']['score_by_category']['final_score']:.1f}")
                    
                    with score_col2:
                        st.markdown("**After Optimization**")
                        st.metric("Match Percentage", f"{score_report['after']['match_percent']}%")
                        st.metric("Final Score", f"{score_report['after']['score_by_category']['final_score']:.1f}")
                    
                    # Add detailed category scores
                    st.markdown("### Detailed Category Scores")
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown("**Before:**")
                        for category, score in score_report['before']['score_by_category'].items():
                            if category != 'final_score':
                                st.metric(category.replace('_', ' ').title(), f"{score:.1f}")
                    
                    with detail_col2:
                        st.markdown("**After:**")
                        for category, score in score_report['after']['score_by_category'].items():
                            if category != 'final_score':
                                st.metric(category.replace('_', ' ').title(), f"{score:.1f}")
                    
                else:
                    st.error("Failed to optimize resume. Please try again.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Add this after displaying the enhanced resume
# === EXPORT SECTION (after Enhanced Resume is shown) ===
if st.session_state.enhanced_resume:
    st.subheader("Export Options")

    # Optional manual contact override
    if st.checkbox("Manually edit contact info", value=False):
        with st.expander("Contact Information for Resume Header", expanded=True):
            st.info("This information will appear in your resume header. At minimum, provide your name.")
            contact_col1, contact_col2 = st.columns(2)

            with contact_col1:
                st.session_state.contact_info['name'] = st.text_input("Full Name", value=st.session_state.contact_info.get('name', ''), key="contact_name")
                st.session_state.contact_info['email'] = st.text_input("Email", value=st.session_state.contact_info.get('email', ''), key="contact_email")
                st.session_state.contact_info['phone'] = st.text_input("Phone", value=st.session_state.contact_info.get('phone', ''), key="contact_phone")
                st.session_state.contact_info['location'] = st.text_input("Location", value=st.session_state.contact_info.get('location', ''), key="contact_location", help="City, State or Country")

            with contact_col2:
                st.session_state.contact_info['linkedin'] = st.text_input("LinkedIn URL", value=st.session_state.contact_info.get('linkedin', ''), key="contact_linkedin")
                st.session_state.contact_info['github'] = st.text_input("GitHub URL", value=st.session_state.contact_info.get('github', ''), key="contact_github")
                st.session_state.contact_info['website'] = st.text_input("Personal Website", value=st.session_state.contact_info.get('website', ''), key="contact_website")

    # Export format + style
    format_col1, format_col2 = st.columns(2)
    with format_col1:
        st.markdown("**Resume Style**")
        resume_style = st.selectbox("Choose formatting style:", options=get_available_styles(), index=0, key="resume_style")
        if resume_style == "ATS-Friendly":
            st.info("Simple, clean format optimized for Applicant Tracking Systems.")
        elif resume_style == "Modern":
            st.info("Contemporary design with clean spacing and subtle color accents.")
        elif resume_style == "Professional":
            st.info("Traditional serif fonts with formal formatting for conservative industries.")

    with format_col2:
        st.markdown("**File Format**")
        export_format = st.radio("Choose export format:", ["PDF", "DOCX", "TXT"], index=0, horizontal=True, key="export_format")
        if export_format == "PDF":
            with st.expander("PDF Generation Troubleshooting"):
                st.markdown("""
                **Note:** PDF generation requires wkhtmltopdf to be installed on your system.

                If you're having issues:
                1. Make sure [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html) is installed
                2. If installed but not found, specify the path below
                """)
                wkhtmltopdf_path = st.text_input("Path to wkhtmltopdf (optional):", placeholder="e.g. C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe", key="wkhtmltopdf_path")
                if wkhtmltopdf_path:
                    import resume_export
                    resume_export.WKHTMLTOPDF_PATH = wkhtmltopdf_path

    # Download section
    export_col1, export_col2, export_col3 = st.columns([1, 2, 1])
    with export_col2:
        if export_format == "PDF":
            try:
                from resume_export import ENABLE_PDF_GENERATION
                if ENABLE_PDF_GENERATION:
                    pdf_bytes = generate_pdf(
                        st.session_state.enhanced_resume,
                        style=resume_style,
                        name=st.session_state.contact_info.get('name', ''),
                        email=st.session_state.contact_info.get('email'),
                        phone=st.session_state.contact_info.get('phone'),
                        location=st.session_state.contact_info.get('location'),
                        linkedin=st.session_state.contact_info.get('linkedin'),
                        github=st.session_state.contact_info.get('github'),
                        website=st.session_state.contact_info.get('website')
                    )
                    st.download_button(f"Download {resume_style} Resume as PDF", pdf_bytes, f"enhanced_resume_{resume_style.lower().replace('-', '_')}.pdf", mime="application/pdf", use_container_width=True)
                else:
                    raise ImportError
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.download_button("Download as Text (Fallback)", st.session_state.enhanced_resume, "enhanced_resume.txt", mime="text/plain", use_container_width=True)

        elif export_format == "DOCX":
            try:
                docx_bytes = generate_docx(
                    st.session_state.enhanced_resume,
                    style=resume_style,
                    name=st.session_state.contact_info.get('name', ''),
                    email=st.session_state.contact_info.get('email'),
                    phone=st.session_state.contact_info.get('phone'),
                    location=st.session_state.contact_info.get('location'),
                    linkedin=st.session_state.contact_info.get('linkedin'),
                    github=st.session_state.contact_info.get('github'),
                    website=st.session_state.contact_info.get('website')
                )
                st.download_button(f"Download {resume_style} Resume as DOCX", docx_bytes, f"enhanced_resume_{resume_style.lower().replace('-', '_')}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
            except Exception as e:
                st.error(f"Error generating DOCX: {e}")
                st.download_button("Download as Text (Fallback)", st.session_state.enhanced_resume, "enhanced_resume.txt", mime="text/plain", use_container_width=True)

        else:  # TXT fallback
            st.download_button("Download as Plain Text", st.session_state.enhanced_resume, "enhanced_resume.txt", mime="text/plain", use_container_width=True)


# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit") 