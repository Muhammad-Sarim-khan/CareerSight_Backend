from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import base64
import io
import json
import google.generativeai as genai
from PIL import Image
import pdf2image
import traceback


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Allow CORS for frontend (adjust origins if deploying)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prompts ---
PROMPTS = {
    "resume_review": """
You are an experienced Technical Human Resource Manager, your task is to review the provided resume against the job description.
Please share your professional evaluation on whether the candidate's profile aligns with the role.
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
""",
    "skills_improvement": """
You are a career coach. Based on the resume and job description, suggest key technical and soft skills the candidate should improve or learn.
Also recommend resources and tools to achieve these improvements.
""",
    "percentage_match": """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
""",
    "linkedin_tips": """
You are a LinkedIn profile optimization expert.
Analyze the provided LinkedIn profile URL and give professional, actionable tips to make it more appealing to recruiters.
Suggest improvements in headline, summary, experience, skills, and endorsements.
Also include tips for SEO optimization so it ranks higher in LinkedIn searches.
""",
    "resume_rewrite": """
You are an expert resume writer. Rewrite the given resume to better match the provided job description.
Provide 2-3 optimized versions focusing on clarity, keyword optimization for ATS, and impact-driven bullet points.
""",
    "job_role_expansion": """
You are a career advisor. Based on the candidate's resume, suggest possible alternative job titles and industries 
they can apply to, highlighting transferable skills and growth opportunities.
""",
    "resume_scoring_dashboard": """
You are an ATS-like resume scoring system. Score the resume against the given job description across multiple 
categories: Skills Match, Experience Relevance, Education Fit, and Keyword Optimization. Provide a JSON-formatted 
output with scores (0-100) for each category and an overall score.
"""
}

# --- Utilities ---
def convert_pdf_to_base64_image(uploaded_file: UploadFile):
    pdf_bytes = uploaded_file.file.read()
    images = pdf2image.convert_from_bytes(pdf_bytes)
    first_page = images[0]

    img_byte_arr = io.BytesIO()
    first_page.save(img_byte_arr, format='JPEG')
    img_data = img_byte_arr.getvalue()

    return base64.b64encode(img_data).decode()


import json
import re

def generate_gemini_response(job_description, image_base64, prompt_type):
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = PROMPTS.get(prompt_type)
    if not prompt:
        raise ValueError("Invalid prompt type")

    pdf_content = [{
        "mime_type": "image/jpeg",
        "data": image_base64
    }]

    def extract_text_from_response(resp):
        try:
            return resp.candidates[0].content.parts[0].text
        except (AttributeError, IndexError):
            raise ValueError("Gemini did not return any text response")

    def extract_json_from_text(text):
        # Try to find JSON block in the text
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                raise ValueError("Gemini returned invalid JSON structure")
        raise ValueError("No JSON object found in Gemini's response")

    # Special handling for resume scoring dashboard
    if prompt_type == "resume_scoring_dashboard":
        prompt += (
            "\nReturn the result ONLY in JSON format like this: "
            '{"resume_score": 85, "category_details": {"skills": "...", "experience": "..."}}'
        )
        response = model.generate_content([prompt, job_description, pdf_content[0]])
        text_output = extract_text_from_response(response)
        return extract_json_from_text(text_output)

    # Default for all other prompts (plain text)
    response = model.generate_content([prompt, job_description, pdf_content[0]])
    return extract_text_from_response(response)


def generate_linkedin_tips(linkedin_url: str):
    """Generate LinkedIn optimization tips using Gemini API."""
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = PROMPTS["linkedin_tips"] + f"\nHere is the LinkedIn profile URL: {linkedin_url}\n"
    response = model.generate_content(prompt)

    return response.text

# --- Unified Route ---
@app.post("/analyze_resume")
async def analyze_resume(
    job_description: str = Form(None),
    prompt_type: str = Form(...),
    resume: UploadFile = File(None),
    linkedin_url: str = Form(None)
):
    try:
        # Handle LinkedIn tips separately
        if prompt_type == "linkedin_tips":
            if not linkedin_url or "linkedin.com/in/" not in linkedin_url:
                return JSONResponse(status_code=400, content={"error": "Invalid LinkedIn profile URL"})
            ai_response = generate_linkedin_tips(linkedin_url)
            return JSONResponse(content={"result": ai_response})

        # All resume-related functionalities
        if not resume:
            return JSONResponse(status_code=400, content={"error": "Resume file is required for this analysis type"})

        image_base64 = convert_pdf_to_base64_image(resume)
        ai_response = generate_gemini_response(job_description or "", image_base64, prompt_type)

        return JSONResponse(content={"result": ai_response})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- New Independent Routes ---

@app.post("/resume_rewrite")
async def resume_rewrite(job_description: str = Form(...), resume: UploadFile = File(...)):
    try:
        image_base64 = convert_pdf_to_base64_image(resume)
        ai_response = generate_gemini_response(job_description, image_base64, "resume_rewrite")
        return JSONResponse(content={"result": ai_response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/job_role_expansion")
async def job_role_expansion(resume: UploadFile = File(...)):
    try:
        image_base64 = convert_pdf_to_base64_image(resume)
        ai_response = generate_gemini_response("", image_base64, "job_role_expansion")
        return JSONResponse(content={"result": ai_response})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/resume_scoring_dashboard")
async def resume_scoring_dashboard(job_description: str = Form(...), resume: UploadFile = File(...)):
    try:
        print("📄 Received Job Description:", job_description[:100])  # first 100 chars
        print("📎 Received File:", resume.filename)

        # Step 1 - Convert PDF
        try:
            image_base64 = convert_pdf_to_base64_image(resume)
            print("✅ PDF converted to base64 image")
        except Exception as e:
            print("❌ PDF conversion failed:", str(e))
            print(traceback.format_exc())
            return JSONResponse(status_code=500, content={"error": f"PDF conversion failed: {str(e)}"})

        # Step 2 - Call Gemini
        try:
            ai_response = generate_gemini_response(job_description, image_base64, "resume_scoring_dashboard")
            print("✅ Gemini API responded")
        except Exception as e:
            print("❌ Gemini call failed:", str(e))
            print(traceback.format_exc())
            return JSONResponse(status_code=500, content={"error": f"Gemini API failed: {str(e)}"})

        return JSONResponse(content={"result": ai_response})

    except Exception as e:
        print("❌ Unknown error:", str(e))
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})