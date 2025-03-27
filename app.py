import streamlit as st
import pdfplumber
import spacy
import openai
from dotenv import load_dotenv
import os
load_dotenv()

# Load NLP Model for Financial Data Extraction
nlp = spacy.load("en_core_web_sm")

# OpenAI API Key 
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to Extract Text from PDF
def extract_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "

    return full_text

# Function to Filter Text for Stock-Related Insights
def filter_stock_related_text(text):
    stock_keywords = [
        "revenue", "earnings", "profit", "loss", "cash flow", "valuation", "shares",
        "Spotify", "subscription", "growth", "market", "competition", "P/E ratio",
        "stock price", "risk", "business model"
    ]
    
    filtered_sentences = [
        sentence for sentence in text.split(". ") if any(keyword in sentence for keyword in stock_keywords)
    ]
    
    return ". ".join(filtered_sentences)

# Function to Use GPT-4 for Investment Insights Extraction
def summarize_stock_pitch(text):
    prompt = f"""
    Extract key financial insights and stock-related details from the following text.
    Focus on revenue, growth, risks, competition, valuation, stock performance, and market analysis for Spotify.

    Text:
    {text}

    Extracted Insights:
    """
    
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=500
    )
    
    return response.choices[0].text.strip()

# Function to Extract Financial Data Using NLP
def extract_financial_entities(text):
    doc = nlp(text)
    relevant_labels = {"MONEY", "ORG", "DATE", "PERCENT", "CARDINAL"}  
    
    entities = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in relevant_labels}
    
    return entities

# Streamlit UI
st.set_page_config(page_title="Spotify Stock Pitch Extractor", layout="wide")
st.title("📊 Spotify Stock Pitch Extractor")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.subheader("📄 Extracting Text...")
    raw_text = extract_pdf_text(uploaded_file)
    filtered_text = filter_stock_related_text(raw_text)
    
    if filtered_text:
        st.subheader("🎯 Filtered Stock Insights")
        st.text_area("Filtered Text", filtered_text, height=300)

        # Summarization
        st.subheader("📌 AI-Generated Stock Insights")
        summary = summarize_stock_pitch(filtered_text)
        st.text_area("Stock Pitch Summary", summary, height=200)

        # Extract Financial Entities
        st.subheader("💰 Financial Data Extracted")
        entities = extract_financial_entities(filtered_text)

        if entities:
            for entity, label in entities.items():
                st.write(f"**{entity}** - {label}")
        else:
            st.write("No financial entities found.")

        # Copy to Clipboard Button
        st.button("📋 Copy Summary to Clipboard", on_click=lambda: st.write(summary))
    else:
        st.warning("No stock-related information found in this PDF.")
