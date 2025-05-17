import streamlit as st
import os
import io
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
import html
import re

# Load environment variables
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Warning: Error configuring Google Gemini API at startup: {e}")
else:
    pass


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file_bytes):
    try:
        pdf_document = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        if not text.strip():
            return None, "No text found in PDF. The PDF might be image-based or empty."
        return text, None
    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"

def generate_credit_analysis_prompt(report_text):
    prompt = f"""
    **Task:** Analyze the following extracted credit report text.
    **Role:** Act as a financial expert.
    **Objective:** Provide an estimated credit score range and a brief analysis of key factors.

    **Instructions:**
    1.  Based ONLY on the information provided in the text below, estimate a credit score range (e.g., 650-680).
    2.  Briefly list the key positive factors influencing this estimate.
    3.  Briefly list the key negative factors influencing this estimate.
    4.  Be concise and structure your answer clearly with headings for each section.
    5.  Do not ask for more information. Do not invent information not present in the text.
    6.  If the text is insufficient, clearly not a credit report, or unanalyzable, state that an analysis cannot be performed and why.

    **Credit Report Text (extract):**
    ---
    {report_text[:30000]}
    ---

    **Expected Output Format:**

    **Estimated Score Range:**
    [Your Estimated Range Here]

    **Positive Factors:**
    - [Factor 1]
    - [Factor 2]
    - ...

    **Negative Factors:**
    - [Factor 1]
    - [Factor 2]
    - ...
    """
    return prompt

def call_gemini_api(prompt_text, model_name="gemini-1.5-flash-latest"):
    if not GOOGLE_API_KEY:
        return None, "Google API key not configured. Please set it in your .env file or deployment secrets."
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_text)

        if response.parts:
            return response.text, None
        else:
            finish_reason_value = "Unknown"
            if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                 finish_reason_value = str(response.prompt_feedback.block_reason)
            if finish_reason_value != "0" and "SAFETY" in finish_reason_value.upper() :
                return None, f"Gemini API content generation stopped due to safety settings (Reason: {finish_reason_value}). Try a different file or contact support if this seems incorrect."
            else:
                return None, f"Gemini API returned an empty response. The content might have been blocked (Reason: {finish_reason_value}) or the prompt was problematic."
    except Exception as e:
        error_detail = str(e)
        if "API key not valid" in error_detail: error_detail = "Google API key not valid. Please check your key."
        elif "quota" in error_detail.lower() or "rate limit" in error_detail.lower(): error_detail = "Google API quota or rate limit exceeded."
        elif "DeadlineExceeded" in error_detail: error_detail = "The request to Google API timed out. Please try again."
        return None, f"Error with Google Gemini API: {error_detail}"

def format_ai_response_for_html(text):
    if not text:
        return ""

    # 1. Escape the entire text initially to handle any stray HTML characters safely.
    processed_text = html.escape(text)

    # 2. Convert Markdown-style bold (**text**) to <strong>text</strong>.
    #    Since the text is already escaped, '**' will be '**'.
    #    We need to match these escaped versions or do bolding before full escape.
    #    Let's do bolding on the original text, then escape.
    
    # Temporarily store original text for bolding
    temp_text_for_bolding = text 
    
    # Bolding: Replace **text** with <strong>html.escape(text)</strong>
    def bold_replacer(match):
        return f"<strong>{html.escape(match.group(1))}</strong>"
    processed_text = re.sub(r"\*\*(.*?)\*\*", bold_replacer, temp_text_for_bolding, flags=re.DOTALL)

    # 3. Color the score range. This regex is more general.
    #    It looks for "Estimated Score Range:" (possibly bolded or not)
    #    followed by a space, then the score "XXX-YYY".
    #    It assumes the bolding of "Estimated Score Range:" has already happened if it was **...**.
    #    The (.*?) before the score is to capture the prefix, including any <strong> tags.
    score_pattern = re.compile(
        r"(Estimated Score Range:(?:<\/strong>)?\s*)(\d{3}\s*-\s*\d{3})",
        re.IGNORECASE
    )
    
    def color_score_replacer(match):
        prefix = match.group(1)  # This includes "Estimated Score Range:" and potentially </strong>
        score_numbers = match.group(2)
        return f'{prefix}<span class="score-value">{html.escape(score_numbers)}</span>'
    
    processed_text = score_pattern.sub(color_score_replacer, processed_text)
    
    # 4. Convert newlines to <br> tags to ensure spacing is preserved in HTML.
    #    This makes it less dependent on pre-wrap for newlines if it's behaving unexpectedly.
    processed_text = processed_text.replace("\n", "<br>\n")

    return processed_text

# --- Streamlit UI ---
st.set_page_config(page_title="JaronAI - Credit Score Checker", layout="wide")

chatgpt_light_theme_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"], .main {font-family: 'Inter', sans-serif; background-color: #FFFFFF; color: #0D0D0D;}
    .title-container {display: flex; flex-direction: column; align-items: center; margin-bottom: 1rem;}
    .main-title {font-family: 'Inter', sans-serif !important; color: #10A37F !important; text-align: center; font-size: 2.8rem !important; font-weight: 700 !important; margin-top: 0.5rem; margin-bottom: 0.1rem;}
    .subtitle {font-family: 'Inter', sans-serif; color: #555555; text-align: center; font-size: 1.1rem; margin-bottom: 2.5rem;}
    .stButton>button {font-family: 'Inter', sans-serif; border: 1px solid #D1D5DB; color: #374151; background-color: #FFFFFF; padding: 0.5rem 1.2rem; border-radius: 0.375rem; transition: all 0.2s ease-in-out; font-weight: 500;}
    .stButton>button:hover {background-color: #F9FAFB; border-color: #9CA3AF;}
    .stButton>button:focus {box-shadow: 0 0 0 0.2rem rgba(16, 163, 127, 0.25) !important; border-color: #10A37F !important;}
    .stFileUploader > label {font-family: 'Inter', sans-serif; color: #374151; font-weight: 500; font-size: 0.95rem;}
    div[data-testid="stFileUploader"] section {background-color: #F9FAFB; border-radius: 0.375rem; border: 1px dashed #D1D5DB;}
    div[data-testid="stFileUploader"] section:hover {border-color: #9CA3AF;}

    /* --- Custom AI Response Box Styling --- */
    .ai-response-box {
        background-color: #F3F4F6 !important; color: #1F2937 !important;
        padding: 1em !important; border-radius: 0.375rem !important;
        border: 1px solid #E5E7EB !important;
        font-family: 'Roboto Mono', monospace !important; font-size: 0.9rem !important;
        /* white-space: pre-wrap !important; /* We use <br> now, but pre-wrap is good for wrapping long lines */
        /* word-wrap: break-word !important; overflow-wrap: break-word !important; */
        /* The above are less critical if <br> handles newlines and the container width is managed */
        line-height: 1.5 !important; /* Improves readability of multi-line text */
        opacity: 1 !important; transition: none !important;
        max-width: 100% !important; box-sizing: border-box !important;
        text-align: left !important; /* Ensure text is aligned left */
    }
    .ai-response-box:hover {background-color: #F3F4F6 !important; color: #1F2937 !important; opacity: 1 !important;}
    
    .ai-response-box .score-value {color: #10A37F !important; font-weight: bold !important;}
    .ai-response-box strong {font-weight: bold !important; color: inherit !important;}
    /* --- End of Custom AI Response Box Styling --- */

    .stAlert {font-family: 'Inter', sans-serif;border-radius: 0.375rem;border-width: 1px;border-left-width: 4px !important;}
    div[data-testid="stAlert"][data-baseweb="alert"][role="alert"].st-emotion-cache-l9qbdf {border-left-color: #3B82F6 !important; background-color: #EFF6FF !important;}
    div[data-testid="stAlert"][data-baseweb="alert"][role="alert"].st-emotion-cache-l9qbdf p { color: #1E40AF !important; }
    div[data-testid="stAlert"][data-baseweb="alert"][role="alert"].st-emotion-cache-okbr68 {border-left-color: #EF4444 !important; background-color: #FEF2F2 !important;}
    div[data-testid="stAlert"][data-baseweb="alert"][role="alert"].st-emotion-cache-okbr68 p { color: #991B1B !important; }
    div[data-testid="stAlert"][data-baseweb="alert"][role="alert"].st-emotion-cache-j7qwkr {border-left-color: #F59E0B !important; background-color: #FFFBEB !important;}
    div[data-testid="stAlert"][data-baseweb="alert"][role="alert"].st-emotion-cache-j7qwkr p { color: #92400E !important; }
    [data-testid="stSidebar"] {background-color: #F9FAFB;border-right: 1px solid #E5E7EB;}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li { color: #4B5563; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 { color: #1F2937; }
    [data-testid="stSidebar"] a { color: #10A37F; }
    .main .block-container {max-width: 800px;padding-left: 1rem;padding-right: 1rem;margin: auto;}
</style>
"""
st.markdown(chatgpt_light_theme_css, unsafe_allow_html=True)

# --- UI Elements ---
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
logo_path = "logo.png"
if os.path.exists(logo_path):
    _, col_logo, _ = st.columns([2,1,2])
    with col_logo: st.image(logo_path, width=80)
else: st.sidebar.warning("Logo file (logo.png) not found in the project directory.")

st.markdown("<div class='title-container'><h1 class='main-title'>Jaron's AI</h1><p class='subtitle'>The FREE credit score checker</p></div>", unsafe_allow_html=True)
st.markdown("**Disclaimer:** This tool provides an AI-generated ESTIMATE. It is NOT official financial advice. Files are NOT stored. This service is for informational purposes only. PDF files only.")
st.sidebar.header("Information")
st.sidebar.info("This web app analyzes uploaded PDF credit reports and provides an estimated credit score range and key financial factors...") # Truncated for brevity
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [Google Gemini](https://ai.google.dev/) & [Streamlit](https://streamlit.io)")

if not GOOGLE_API_KEY:
    st.error("FATAL ERROR: Google API Key is not configured...") # Truncated
    st.stop()

uploaded_file = st.file_uploader("Upload your Credit Report (PDF only)", type=["pdf"], help="Upload a text-based PDF file...")
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    st.success(f"File selected: **{file_name}**")
    if st.button("Analyze PDF with AI", key="analyze_button", use_container_width=True):
        with st.spinner("Extracting text and analyzing PDF..."):
            extracted_text, extraction_error = extract_text_from_pdf(file_bytes)
            if extraction_error: st.error(f"PDF Processing Error: {extraction_error}")
            elif extracted_text:
                analysis_prompt = generate_credit_analysis_prompt(extracted_text)
                ai_response, ai_error = call_gemini_api(analysis_prompt)
                if ai_error: st.error(f"AI Analysis Error: {ai_error}")
                elif ai_response:
                    st.subheader("AI Analysis Results:")
                    formatted_html_response = format_ai_response_for_html(ai_response)
                    # Use a <div> container now since we are inserting <br> tags
                    # The <pre> was mainly for its white-space: pre-wrap behavior.
                    # If <br> handles newlines, a div is fine and might be less semantically confusing.
                    # However, <pre> also implies monospaced font and preformatted text, which we still want.
                    # So let's stick with <pre> and ensure its `white-space` doesn't conflict with our <br>s.
                    # `white-space: pre-wrap` on the <pre> will wrap long lines that don't have <br>,
                    # and it will also respect the <br> tags.
                    response_html_container = f"""
                    <pre class="ai-response-box">{formatted_html_response}</pre>
                    """
                    st.markdown(response_html_container, unsafe_allow_html=True)
                # else: (Handled by ai_error or if ai_response is None/empty without specific error)
            else: st.warning("No text could be extracted...") # Truncated
        if not extraction_error and not ai_error and not ai_response and uploaded_file: # Check if analysis_requested was true implicitly
             st.warning("Analysis could not be completed...") # Truncated
else: st.info("Upload a PDF file to begin.")