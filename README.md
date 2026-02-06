# AI Document Orchestrator

Streamlit app that extracts text from PDFs or TXT files, performs dynamic structured extraction using OpenRouter models with optional RAG (retrieval + embeddings), and triggers an n8n webhook for conditional email automation.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your secrets in `.streamlit/secrets.toml`:

```
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
N8N_WEBHOOK_URL = "YOUR_N8N_PROD_WEBHOOK_URL"
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"  # optional
OPENROUTER_TIMEOUT = 60  # optional
OPENROUTER_RETRIES = 2   # optional
```

## Run

```bash
streamlit run app.py
```

## RAG (Retrieval + Embeddings)

When enabled in the sidebar, the app:

1. Chunks the extracted document text.
2. Creates embeddings for each chunk.
3. Retrieves the top‑K most relevant chunks for the user question.
4. Uses only those chunks as evidence for structured extraction.

This improves precision and helps the model focus on the most relevant clauses and metrics.

## n8n Workflow

Use the following node order and logic:

1. Webhook Trigger (POST, response mode: Using Respond to Webhook)
2. AI Agent - Final Analysis
3. IF Node: `{{$json.structured_data.risk_level}} = High`
4. TRUE branch: AI Agent - Email Draft -> Email Node (Gmail / SMTP) -> Set Node
5. FALSE branch: Set Node (email_body, status)
6. Respond to Webhook (return JSON with final_answer, email_body, status)

## n8n Setup Steps (v2.6.3)

1. Create a new workflow named `AI Document Orchestrator`.
2. Add **Webhook** node:
   - Method: `POST`
   - Respond: `Using Respond to Webhook`
   - Copy **Production URL** and set `N8N_WEBHOOK_URL` in `.streamlit/secrets.toml`.
3. Add **AI Agent** node (Final Analysis):
   - Prompt:
     ```
     Using the document text, structured data, and user question,
     produce a concise analytical answer.
     ```
   - Prompt must inject values (Expression mode):
     ```
     DOCUMENT TEXT:
     {{$json.body.document_text}}

     STRUCTURED DATA:
     {{ JSON.stringify($json.body.structured_data) }}

     USER QUESTION:
     {{$json.body.question}}
     ```
4. Add **IF** node:
   - Condition (Expression):
     ```
     {{$node["Webhook"].json.body.structured_data["Risk Level"]}}
     ```
     equals `High`
   - If your JSON uses `risk_level`, switch to:
     ```
     {{$node["Webhook"].json.body.structured_data.risk_level}}
     ```
5. TRUE branch:
   - **AI Agent** (Email Draft) prompt:
     ```
     Draft a professional alert email explaining the risk and recommended action.

     STRUCTURED DATA:
     {{ JSON.stringify($json.body.structured_data) }}

     FINAL ANSWER:
     {{$json.output}}
     ```
   - **Gmail** node: Send a message
     - To: `{{$json.body.recipient_email}}`
     - Message: map to the email draft output (for example `{{$json.output}}`)
   - **Set** node:
     - `final_answer` = `{{$json.output}}` (from the Final Analysis AI Agent)
     - `email_body` = `{{$json.output}}` (from Email Draft AI Agent)
     - `status` = `Alert Email SENT`
6. FALSE branch:
   - **Set** node:
     - `final_answer` = `{{$json.output}}`
     - `email_body` = `Condition not met. Email not sent.`
     - `status` = `Condition Not Met`
7. Add **Respond to Webhook** node:
   - Response Body:
     ```
     {
       "final_answer": "={{$json.final_answer}}",
       "email_body": "={{$json.email_body}}",
       "status": "={{$json.status}}"
     }
     ```
8. Publish the workflow, then test via Streamlit.

## Payload Notes

When RAG is enabled, the app also sends `retrieved_chunks` to n8n for logging or auditing. You can ignore this field if not needed.
