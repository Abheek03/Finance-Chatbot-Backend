from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# It is recommended to use an environment variable for the API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------
# ✅ CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 📊 LOAD + CLEAN DATA
# ---------------------------
def load_and_prepare_data():
    df = pd.read_excel("cards_transactions_dataset.xlsx")

    # Correct column names based on user's data snippet
    df.columns = ['transaction_id', 'customer_id', 'amount', 'card_type', 'card_network', 'merchant_category', 'region', 'customer_age', 'transaction_date']

    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df = df.dropna()
    df = df.drop_duplicates()
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Feature engineering
    df['month'] = df['transaction_date'].dt.to_period('M')
    df['year'] = df['transaction_date'].dt.year
    df['day_of_week'] = df['transaction_date'].dt.day_name()
    df['quarter'] = df['transaction_date'].dt.to_period('Q')

    df['age_group'] = pd.cut(
        df['customer_age'],
        bins=[18, 25, 35, 50, 65, 100],
        labels=['18-25', '26-35', '36-50', '51-65', '65+']
    )

    df['is_high_value'] = df['amount'] > df['amount'].quantile(0.75)

    return df

df = load_and_prepare_data()

# ---------------------------
# 🔐 SAFE EXECUTION
# ---------------------------
# WARNING: Executing code from user input is extremely dangerous and should not be used in a production environment.
# A malicious user could craft a question that generates code to access sensitive information or harm your system.
def execute_code(code, df):
    try:
        safe_globals = {"__builtins__": {}}
        safe_locals = {"df": df, "pd": pd}

        exec(code, safe_globals, safe_locals)

        return safe_locals.get("result", "No result generated")
    except Exception as e:
        return str(e)

# ---------------------------
# 🧠 STEP 1: Generate Pandas Code
# ---------------------------
def generate_pandas_code(question, previous_code=None, error=None):
    system_message = f"""You are a data analyst.

Convert the question into VALID Python Pandas code.

DataFrame name: df
Available columns: {list(df.columns)}

Your job:
- Convert question into pandas code
- Use ONLY available columns
- Handle missing columns gracefully
- NEVER use .str on datetime columns
- Use df['transaction_date'].dt.year for year filtering
- For monthly trends, use df['transaction_date'].dt.to_period('M'). For quarterly trends, use df['transaction_date'].dt.to_period('Q').
- For time-based comparisons, use `pct_change()` or filter data for specific time ranges.
- For correlation analysis, use the `.corr()` method.
- For customer segmentation, use the `customer_id` column.

Columns:
- transaction_id
- customer_id
- amount
- card_type (credit_card / debit_card)
- card_network (Visa / Mastercard)
- merchant_category
- region
- customer_age
- transaction_date (datetime)
- year
- day_of_week
- age_group
- is_high_value

STRICT RULES:
- Use pandas only
- DO NOT import anything
- DO NOT use print()
- DO NOT generate any code for plotting or visualization.
- transaction_date is already datetime → use .dt.year
- ALWAYS write syntactically correct code
- ALWAYS use proper dots (.)
- NO string operations on dates
- ONLY code
- Store final output in a variable called 'result'
- If the question is ambiguous, vague, or a simple greeting (e.g., "hi", "hello", "sample"), return: result = "AMBIGUOUS_QUERY"
- If unsure about how to answer, return: result = "NO_VALID_DATA"
- Provide no explanation, only code.

EXAMPLE:
Question: In 2024 which merchant category had highest sales?

result = df[df['transaction_date'].dt.year == 2024] \\
    .groupby('merchant_category')['amount'] \\
    .sum() \\
    .sort_values(ascending=False) \\
    .head(1)

Question: What is the monthly trend of sales?

result = df.groupby(df['transaction_date'].dt.to_period('M'))['amount'].sum()

Question: What is the month-on-month revenue growth rate?

result = df.groupby(df['transaction_date'].dt.to_period('M'))['amount'].sum().pct_change()

Question: Is there a correlation between customer age and transaction amount?

result = df[['customer_age', 'amount']].corr()

Question: Who are the top 5 customers by total spending?

result = df.groupby('customer_id')['amount'].sum().nlargest(5)
"""

    if previous_code and error:
        system_message += f"""
The user's previous attempt to generate code failed.
Previous code:
{previous_code}

Error:
{error}

Please analyze the error and the previous code, and generate a new, corrected version of the pandas code to answer the user's question.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {"role": "user", "content": question}
        ]
    )

    code = response.choices[0].message.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    print("Generated Code:\n", code)

    return code

# ---------------------------
# 📊 STEP 2: Format Output
# ---------------------------
def format_response(result):
    if isinstance(result, (int, float)):
        return {"answer": f"The result is {result}.", "data": [{"result": result}]}

    if isinstance(result, pd.Series):
        result = result.reset_index()

    if isinstance(result, pd.DataFrame):
        # Replace NaN with None for JSON serialization
        result.replace({np.nan: None}, inplace=True)

        # Convert period/datetime to string for JSON serialization
        if pd.api.types.is_datetime64_any_dtype(result.iloc[:, 0]) or pd.api.types.is_period_dtype(result.iloc[:, 0]):
            result.iloc[:, 0] = result.iloc[:, 0].astype(str)

        data = result.to_dict(orient="records")
        answer = "Here is the data you requested." # Generic answer

        if not data:
            answer = "No data found for this query."
        elif len(data) == 1:
            answer = "Here is the single record I found."
        
        return {"answer": answer, "data": data}

    return {"answer": str(result), "data": []}

# ---------------------------
# 💼 STEP 3: Business Answer
# ---------------------------
def generate_business_answer(question, formatted, pandas_code):
    # Check if there is no data
    if not formatted['data']:
        return "I couldn't find any data for your query. It's possible that the data for the requested period or category does not exist. Please try a different query."

    # Check if the data is a time series
    if len(formatted['data']) > 1 and ('month' in formatted['data'][0] or 'quarter' in formatted['data'][0]):
        # This is a trend query
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are a financial business analyst. Your task is to provide a comprehensive analysis of the provided data in response to the user's question.

- Analyze the provided time series data and summarize the trend.
- Include the value for each period (e.g., each month or quarter) in your summary.
- Identify the period with the highest and lowest values.
- Provide a concise and to-the-point summary of the trend.
- Add one small, actionable insight based on the result.
- Do not provide technical explanations about the code.
- Your response should be in a clear and easy-to-understand format.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Question: {question}
Pandas Code:
{pandas_code}
Data:
{formatted['data']}
"""
                }
            ]
        )
        return response.choices[0].message.content

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a financial business analyst. Your task is to provide a comprehensive analysis of the provided data in response to the user's question.

- Give clear business answers based on the data provided.
- Do not provide technical explanations about the code.
- Be concise and to the point.
- Add one small, actionable insight based on the result.
- Your response should be in a clear and easy-to-understand format.
"""
            },
            {
                "role": "user",
                "content": f"""
Question: {question}
Pandas Code:
{pandas_code}
Base Answer: {formatted['answer']}
Data: {formatted['data']}
"""
            }
        ]
    )

    return response.choices[0].message.content

def clarify_question(question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a data analyst. Your role is to help users clarify their ambiguous questions about financial data.

- The user's question is ambiguous.
- Your task is to ask a clarifying question to help the user get a more specific and useful answer.
- Provide a few options as examples, if possible.
- Keep your question concise and to the point.

Example:
User question: "What are the sales?"
Your clarifying question: "Could you please specify the period for which you want to see the sales data? For example, you can ask for monthly, quarterly, or yearly sales."
"""
            },
            {
                "role": "user",
                "content": f"The user's question is: '{question}'"
            }
        ]
    )
    return response.choices[0].message.content

# ---------------------------
# ✅ Request schema
# ---------------------------
class Query(BaseModel):
    question: str

class ClarifyQuery(BaseModel):
    original_question: str
    user_response: str

# ---------------------------
# 🚀 MAIN API ENDPOINT
# ---------------------------
MAX_RETRIES = 3

@app.post("/chat")
def chat(query: Query):
    try:
        question = query.question
        pandas_code = None
        result = None
        error_message = None
        
        for attempt in range(MAX_RETRIES):
            # Step 1: Generate Pandas code from the user's question
            pandas_code = generate_pandas_code(question, previous_code=pandas_code, error=error_message)
            
            # Handle cases where the query was ambiguous
            if pandas_code == 'result = "AMBIGUOUS_QUERY"':
                clarifying_question = clarify_question(question)
                return {"answer": clarifying_question, "type": "clarification"}

            # Step 2: Execute the generated code
            result = execute_code(pandas_code, df)

            # Check for errors
            if isinstance(result, str):
                error_message = result
                # if it's the last attempt, return the error
                if attempt == MAX_RETRIES - 1:
                    return {"answer": f"An error occurred while processing the data: {result}"}
                # otherwise, continue to the next iteration to retry
                continue

            # If execution is successful, break the loop
            break

        # if after all retries, we still have an error
        if error_message and not isinstance(result, (pd.DataFrame, pd.Series)):
             return {"answer": f"An error occurred while processing the data: {error_message}"}


        if isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
            return {"answer": "I couldn't find any data for your query. Please try rephrasing your question or asking something else."}

        # Step 3: Format the result into a preliminary answer
        formatted = format_response(result)

        # Step 4: Generate a final, business-oriented answer
        final_answer = generate_business_answer(question, formatted, pandas_code)

        return {
            "generated_code": pandas_code,
            "answer": final_answer,
            "data": formatted["data"]
        }

    except Exception as e:
        return {"answer": f"An unexpected error occurred: {str(e)}"}

@app.post("/clarify")
def clarify(query: ClarifyQuery):
    # Combine the original question with the user's response to form a new, more specific question
    new_question = f"{query.original_question} {query.user_response}"
    return chat(Query(question=new_question))