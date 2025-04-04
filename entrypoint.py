# entrypoint.py

from workflow import run_enhancement_pipeline
from environment import setup_environment

if __name__ == "__main__":
    setup_environment()

    resume_file = "docs/sample_resume.pdf"
    job_input = """The Military Sealift Command’s (MSC) Comptroller Directorate (N8) oversees the financial operations, budgeting, and audit support functions that ensure MSC’s fiscal responsibilities are met in alignment with DoD standards. The Financial Data Analyst will provide entry-level administrative and analytical support to the MSC Comptroller Directorate team, with a focus on financial data tracking, management, and analysis, documentation, and system process support. They will be empowered to leverage and hone their foundational data skills (e.g. programming, analytics, automation, etc) to advance the project and client objectives. 

 

The role involves coordination with internal stakeholders, maintaining accurate records, and supporting system-related tasks to enhance operational efficiency and audit readiness. 

 

The Financial Data Analyst will be responsible for the following areas of work: 

Provide administrative and operational support to MSC’s Comptroller Directorate (N8). 
Support the ideation, development, and implementation of technology enablers (e.g. automation, analytics, etc.) 
Assist in organizing and tracking financial and audit-related documentation. 
Maintain internal tracking systems, spreadsheets, and filing logs. 
Support meeting coordination, including scheduling, notetaking, and follow-up. 
Help compile data for reports, briefings, and leadership materials. 
Coordinate with internal stakeholders to assist in completing daily tasks. 
Perform other duties as assigned. 


Requirements: 

Bachelor’s degree in a STEM or business-related field. 
Strong communication and organizational skills. 
Ability to work collaboratively in a team environment. 
Proficient in Microsoft Office Suite (Word, Excel, PowerPoint). 
Foundational experience with data tools and/or programming languages for example (but not limited to) VBA, R, Python, PowerPlatform, SQL, Tableau, PowerBI, etc. 
Ability to manage multiple tasks and deadlines. 


Preferred: 

Internship or academic project experience in a government, financial, or audit-related setting. 
Familiarity with Navy ERP (SAP) or other ERP systems. 
Interest in public service or national security initiatives. 


Eligibility: 

Must be a U.S. Citizen 
Must possess a DoD Secret Security Clearance (Interim minimum). Given lead time, new clearances applications will not be sponsored for this role. 
Must possess a bachelor’s degree in STEM or a business-related field. """

    enhanced = run_enhancement_pipeline(resume_file, job_input)

print("\n=== FINAL ENHANCED RESUME ===\n")

section_order = ["summary", "experience", "skills", "education", "certifications", "projects", "other"]
for name in section_order:
    if name in enhanced:
        print(f"\n--- {name.capitalize()} ---\n")
        print(enhanced[name].strip())
