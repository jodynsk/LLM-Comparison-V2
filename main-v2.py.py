import streamlit as st
import requests
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI 

# Initialize the OpenAI client
client = OpenAI(api_key= "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")
# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Function to get response from Ollama
def get_ollama_response(prompt, model):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling Ollama: {e}"

def call_both_models(question: str):
    llama3_resp = get_ollama_response(question, "llama3.1:8b")
    qwen_resp = get_ollama_response(question, "qwen3:4b")
    return llama3_resp, qwen_resp
# Function to validate ALL answers at once using ChatGPT

def validate_all_with_chatgpt(df):

    items = []
    for _, row in df.iterrows():
        items.append({
            "id": str(row["id"]),
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "answer1": row["llama3_answer"],
            "answer2": row["qwen_answer"],
        })

    # One global prompt with all questions + answers
    prompt = f"""
You are an evaluator for 2 LLM answers.

You will be given a LIST of items. Each item has:
1) A quation ID
2) A question 
2) The expert ground-truth answer (gold standard) 
3) Model 1 answer llama3_answer
4) Model 2 answer qwen_answer

For EACH Model, grade all answers on 5 dimensions, each 0–5:

Accuracy (weight 0.50)
Completeness (weight 0.20)
Reasoning Quality (weight 0.15)
Clarity (weight 0.10)
Safety / No Hallucination (weight 0.05)

Then compute FInal score:
Final = 0.5*Accuracy + 0.2*Completeness + 0.15*Reasoning + 0.1*Clarity + 0.05*Safety

Return STRICT like this :

table list of the qutions and the answers

(The Scores)

Module 1 Deepseek score is:
Accuracy: 5
Completeness: 5
Clarity: 4
Reasoning Quality: 5
Safety / No Hallucination: 4
Final Score: 5

Module 2  Qwen score is:
Accuracy: 5
Completeness: 5
Clarity: 4
Reasoning Quality: 5
Safety / No Hallucination: 4
Final Score: 5

compare the 2 models based on the Final Score and say which one is better overall or if they are tied.

Here is the list of items you must evaluate:
{json.dumps(items, ensure_ascii=False, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" for the bigger model
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content



# Streamlit app
st.title("LLMs Evaluation Platform")
st.subheader("Upload Excel file (questions + ground truth)")


uploaded_file = st.file_uploader(
    "Upload .xlsx file",
    type=["xlsx"]
)

weights_df = pd.DataFrame({
    "Metric": [
        "Accuracy  (How factually correct compared to ground truth)",
        "Completeness  (Does the model cover all required points)",
        "Reasoning Quality  (Is the logic sound and well-explained)",
        "Clarity  (Is the answer understandable and structured)",
        "Safety / No Hallucination  (Does it avoid invented facts)"
    ],
    "Weight (%)": [
        50, 20, 15, 10, 5,
    ]
}).set_index("Metric")

st.subheader("Evaluation Criteria & Weights")
st.caption("The following weighting is used by many academic evaluation frameworks because accuracy must be the strongest factor, but completeness and reasoning still matter.")
st.dataframe(weights_df.style.format("{:.0f}%"))

qa_df = None
if uploaded_file is not None:
    try:
        qa_df = pd.read_excel(uploaded_file)

        # Ensure key columns exist
        required_cols = {"id", "question", "ground_truth"}
        if not required_cols.issubset(set(qa_df.columns)):
            st.error(f"Excel must contain these columns: {required_cols}")
            st.stop()

        # Ensure types
        qa_df["id"] = qa_df["id"].astype(str)
        qa_df["question"] = qa_df["question"].astype(str)
        qa_df["ground_truth"] = qa_df["ground_truth"].astype(str)

        st.success("Excel loaded successfully ✅")
        st.write("Preview:")
        st.dataframe(qa_df.head())

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()
else:
    st.stop()

st.markdown("---")
st.subheader("Run evaluation on **all** questions in the Excel")

if st.button("Run Tool"):
    total_questions = len(qa_df)
    progress_bar = st.progress(0)
    counter_text = st.empty()

    # 1) Ask local models for ALL questions (in parallel)
    with st.spinner("The local models are thinking..."):
        questions_list = qa_df["question"].tolist()
        llama3_answers = []
        qwen_answers = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            for idx, (llama_ans, qwen_ans) in enumerate(
                executor.map(call_both_models, questions_list),
                start=1
            ):
                llama3_answers.append(llama_ans)
                qwen_answers.append(qwen_ans)
                counter_text.markdown(
                f"**Done:** {idx}/{total_questions}"
                 )
                progress_bar.progress(idx / total_questions)

        qa_df["llama3_answer"] = llama3_answers
        qa_df["qwen_answer"] = qwen_answers

    st.success("Local models finished answering all questions ✅")

    st.write("Sample of questions + answers:")
    st.dataframe(
        qa_df[["id", "question", "ground_truth", "llama3_answer", "qwen_answer"]].head()
    )

    # 2) Send ALL answers at once to ChatGPT
    with st.spinner("Wait, we are cooking..."):
        validation_result = validate_all_with_chatgpt(qa_df)

    st.subheader("Evaluation Results:")
    st.markdown(validation_result)
