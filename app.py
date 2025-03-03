import camelot
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import csv
from groq import Groq
from together import Together
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import math
from pathlib import Path
import io
import pickle
import joblib 
from pydantic import BaseModel



load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
SAMPLE_DIR = BASE_DIR / "samples"
SAMPLE_CSV = SAMPLE_DIR / "sample_statement.csv"
WORKING_CSV = BASE_DIR / "working_statement.csv"
PKL_MODEL = BASE_DIR / "model.pkl"

UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

# Initialize Groq client
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Initialize Together client
together_client = Together()

SAMPLE_STATEMENTS = {
    "hdfc_2023": "sample_statements/hdfc_2023.csv",
    # "hdfc_2022": "sample_statements/hdfc_2022.csv",
    # "sbi_2023": "sample_statements/sbi_2023.csv"    
}
def extract_tables(file_path):
    """Extract tables from PDF using Camelot and save raw data to CSV"""
    all_data = []
    tables = camelot.read_pdf(file_path, pages='all')
    print(f"Total tables found: {tables.n}")
    
    for table in tables:
        df = table.df
        all_data.append(df)
    
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("extracted_tables.csv", index=False)
    return final_df



def process_chunk(chunk_data, chunk_index, total_chunks, starting_index=0, use_together=False):
    """Processes a single chunk of data using LLM"""
    
    system_prompt = """You are a financial data assistant. Your task is to clean, extract, and structure raw bank statement data into a well-organized CSV format.
    Return ONLY the CSV content with no additional explanations or markdown.
    The CSV must have the following columns:
    Index,Narration,Date,Date_Formated,Withdrawal,Deposited,Balance,UPI_Name,UPI_Bank,UPI_Description,Cumulative_Withdrawal,Cumulative_Deposited"""
    
    user_prompt = f"""Format the following bank statement data into a structured CSV.
    This is chunk {chunk_index+1} of {total_chunks}.
    Start your Index at {starting_index} to ensure continuity between chunks.
    
    ### **Raw Data to Process**:
    {chunk_data}
    
    ### **Guidelines**:
    - Convert 'Date' to '%d/%m/%y' format and create a new column 'Date_Formated' as '%d-%b-%Y'.
    - Format 'Withdrawal' and 'Deposited' to one decimal place.
    - Convert 'Balance' to float.
    - Extract 'UPI_Name' and 'UPI_Bank' from 'Narration'.
    - Extract 'UPI_Description' using a function.
    - Calculate 'Cumulative_Withdrawal' and 'Cumulative_Deposited'.
    - Do not include any markdown or explanations, just return the CSV content with headers.
    - If this is NOT the first chunk, do not include the header row.
    - Ensure each row has a numeric index and all required fields.
    - Do not include rows that don't contain valid transaction data.
    """
    
    if use_together:
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=2000
        )
    else:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=2000
        )
    
    return response.choices[0].message.content.strip()



def validate_csv(csv_data, expected_columns=12):
    """
    Validates and cleans CSV data to ensure proper formatting.
    - Ensures all rows have the correct number of columns.
    - Preserves quoted multi-line values.
    - Normalizes numeric values by removing commas.
    - Removes rows that don't contain valid transaction data.
    """
    corrected_rows = []
    reader = csv.reader(io.StringIO(csv_data), quotechar='"')
    
    # Get header row
    headers = next(reader, None)
    if headers:
        corrected_rows.append(headers)
    
    for row in reader:
        # Skip rows with only an index and no other data
        if len(row) <= 1 or (len(row) > 1 and all(not cell.strip() for cell in row[1:])):
            continue
            
        # Ensure the row has the correct number of columns
        if len(row) > expected_columns:
            row = row[:expected_columns]  # Trim extra columns
        elif len(row) < expected_columns:
            row.extend([""] * (expected_columns - len(row)))  # Fill missing values
        
        # Check if this is a valid transaction row (has date and either withdrawal, deposit, or balance)
        has_date = any(cell.strip() and '/' in cell for cell in row[2:4])  # Check Date columns
        has_transaction = any(cell.strip() for cell in row[4:7])  # Check transaction amount columns
        
        if not (has_date and has_transaction):
            continue  # Skip invalid rows
        
        # Normalize numeric values (removing commas)
        for i in range(len(row)):
            if any(char.isdigit() for char in row[i]):  # Check if field contains numbers
                row[i] = row[i].replace(",", "")  # Remove thousand separators
        
        corrected_rows.append(row)

    # Convert back to CSV format
    output = io.StringIO()
    writer = csv.writer(output, quotechar='"')
    writer.writerows(corrected_rows)
    return output.getvalue()

def cleanCsv(file_path, chunk_size=200, use_together=False):
    """Reads the extracted CSV in chunks, processes each chunk, and merges results"""
    
    # Read raw extracted CSV
    raw_df = pd.read_csv(file_path)
    
    # Convert DataFrame to string format for processing
    raw_data_str = raw_df.to_csv(index=False)
    
    # Split into chunks
    lines = raw_data_str.split("\n")
    total_chunks = math.ceil(len(lines) / chunk_size)
    
    all_chunks_combined = []
    header = None
    
    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_data = "\n".join(lines[start:end])
        
        print(f"Processing chunk {i+1}/{total_chunks}...")
        # Pass current count of rows as starting index
        starting_index = len(all_chunks_combined)
        cleaned_chunk = process_chunk(chunk_data, i, total_chunks, starting_index, use_together)
        
        # Process chunk into rows
        chunk_rows = cleaned_chunk.strip().split('\n')
        
        # Extract header from first chunk
        if i == 0 and len(chunk_rows) > 0:
            header = chunk_rows[0]
            # Add data rows from first chunk
            if len(chunk_rows) > 1:
                all_chunks_combined.extend(chunk_rows[1:])
        else:
            # For subsequent chunks, skip header if present
            if chunk_rows and chunk_rows[0].startswith("Index,Narration"):
                if len(chunk_rows) > 1:
                    all_chunks_combined.extend(chunk_rows[1:])
            else:
                all_chunks_combined.extend(chunk_rows)
    
    # Make sure we have a valid header
    if not header:
        header = "Index,Narration,Date,Date_Formated,Withdrawal,Deposited,Balance,UPI_Name,UPI_Bank,UPI_Description,Cumulative_Withdrawal,Cumulative_Deposited"
    
    # Combine all chunks with header
    final_csv_content = header + '\n' + '\n'.join(all_chunks_combined)
    
    # Write the raw combined data first
    raw_output_file = "raw_cleaned_statement.csv"
    with open(raw_output_file, "w", encoding="utf-8") as file:
        file.write(final_csv_content)
    
    print(f"Raw combined CSV saved to {raw_output_file}")
    
  
    try:
        # Read the raw combined data
        # Updated parameter from error_bad_lines to on_bad_lines
        df = pd.read_csv(raw_output_file, on_bad_lines='warn')
        
        # Basic cleaning
        # Fill missing values appropriately
        df = df.fillna({
            'Narration': '',
            'UPI_Name': '',
            'UPI_Bank': '',
            'UPI_Description': ''
        })
        
        # Ensure date fields are properly formatted
        # First, check if Date column exists
        if 'Date' in df.columns:
            # Try to convert dates, keeping original if fails
            try:
                # Handle date conversion errors gracefully
                df['Date_Temp'] = pd.to_datetime(df['Date'], 
                                                format='mixed',
                                                dayfirst=True, 
                                                errors='coerce')
                # Keep only rows with valid dates
                df = df.dropna(subset=['Date_Temp'])
                # Format the dates correctly
                df['Date'] = df['Date_Temp'].dt.strftime('%d/%m/%y')
                df['Date_Formated'] = df['Date_Temp'].dt.strftime('%d-%b-%Y')
                # Remove temporary column
                df = df.drop('Date_Temp', axis=1)
            except Exception as e:
                print(f"Date conversion warning: {e}")
        
        # Convert numeric columns
        numeric_cols = ['Withdrawal', 'Deposited', 'Balance', 
                        'Cumulative_Withdrawal', 'Cumulative_Deposited']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove invalid rows
        df = df.dropna(subset=['Date'], how='all')
        
        # Ensure no duplicate index values
        df = df.reset_index(drop=True)
        df['Index'] = df.index
        
        # Save the clean data
        output_file = "cleaned_bank_statement.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Successfully processed and validated {len(df)} rows of data")
        return df
        
    except Exception as e:
        print(f"Error processing combined data: {e}")
        # Emergency fallback - try to recover data manually
        try:
            with open(raw_output_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                raise ValueError("No data found in combined CSV")
            
            header_line = lines[0].strip()
            data_lines = [line for line in lines[1:] if line.strip() and ',' in line]
            
            if not data_lines:
                raise ValueError("No valid data rows found")
            
            # Write clean data
            output_file = "cleaned_bank_statement.csv"
            with open(output_file, 'w') as f:
                f.write(header_line + '\n')
                f.writelines(data_lines)
            
            print(f"Recovered {len(data_lines)} rows using manual recovery")
            
            # Try to read the manually fixed file
            # Updated parameter from error_bad_lines to on_bad_lines
            df = pd.read_csv(output_file, on_bad_lines='warn')
            return df
            
        except Exception as recovery_error:
            print(f"Recovery attempt failed: {recovery_error}")
            # Create minimal valid DataFrame to prevent further errors
            columns = header.split(',')
            df = pd.DataFrame(columns=columns)
            df.to_csv("cleaned_bank_statement.csv", index=False)
            return df
 

@app.get("/sample-statements")
async def get_sample_statements():
    return [
        {
            "id": key,
            "name": key.replace("_", " ").title(),
            "description": f"Sample {key.replace('_', ' ').title()} Statement"
        }
        for key in SAMPLE_STATEMENTS.keys()
    ]

@app.post("/use-sample")
async def use_sample_statement():
    try:
        if not SAMPLE_CSV.exists():
            raise HTTPException(status_code=404, detail="Sample statement not found")
        process_data_for_analysis(SAMPLE_CSV)
        return {"message": "Sample statement processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_pdf = None
    
    try:
        # Ensure directories exist
        UPLOAD_DIR.mkdir(exist_ok=True)
        WORKING_CSV.parent.mkdir(exist_ok=True)
        
        # Save uploaded PDF
        temp_pdf = UPLOAD_DIR / "temp.pdf"
        with temp_pdf.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Extract tables from PDF
        raw_df = extract_tables(str(temp_pdf))
        
        if raw_df.empty:
            raise HTTPException(status_code=400, detail="No valid tables found in PDF")

        # Process with LLM and save result
        cleaned_df = cleanCsv(str("extracted_tables.csv"), use_together=True)
        cleaned_df.to_csv(str(WORKING_CSV), index=False)
        
        # Process data for analysis
        process_data_for_analysis(WORKING_CSV)
        
        return {
            "message": "File processed successfully",
            "rows_processed": len(cleaned_df)
        }
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary files
        if temp_pdf and temp_pdf.exists():
            temp_pdf.unlink()
        if Path("extracted_tables.csv").exists():
            Path("extracted_tables.csv").unlink()

# df = cleanCsv("extracted_tables.csv", use_together=True)
# df = pd.read_csv("cleaned_bank_statement.csv")

@app.get("/summary")
def get_summary():
    # Load Data
    df = pd.read_csv(WORKING_CSV)
    df = df.replace({np.nan: None})
    
    # Convert Date column to datetime (ISO format: YYYY-MM-DD)
    df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d', errors="coerce")

    # Drop invalid dates
    df = df.dropna(subset=["Date"])
    if df.empty:
        return {"error": "No valid date entries found in the dataset."}

    # Extract correct date format for display
    start_date = df['Date'].iloc[0].strftime("%d %B %Y")  # Example: "01 November 2022"
    end_date = df['Date'].iloc[-1].strftime("%d %B %Y")  # Example: "04 November 2022"

    # Calculate duration
    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days + 1  # Ensure at least 1 day

    # Convert necessary columns to numeric
    df["Withdrawal"] = pd.to_numeric(df["Withdrawal"], errors="coerce").fillna(0)
    df["Deposited"] = pd.to_numeric(df["Deposited"], errors="coerce").fillna(0)
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(0)

    total_withdrawal = df["Withdrawal"].sum()
    total_deposit = df["Deposited"].sum()

    # Return Summary
    return {
        "period": {
            "start": start_date,
            "end": end_date,
            "days": days
        },
        "metrics": {
            "totalTransactions": len(df),
            "totalWithdrawal": float(total_withdrawal),
            "totalDeposit": float(total_deposit),
            "closingBalance": float(df["Balance"].iloc[-1]),
            "openingBalance": float(df["Balance"].iloc[0]),
            "avgWithdrawalPerDay": float(total_withdrawal / days) if days else 0,
            "avgWithdrawalPerMonth": float(total_withdrawal / (days / 30)) if days else 0
        }
    }

@app.get("/trends")
async def get_trends():
    try:
        df = pd.read_csv(WORKING_CSV)
        df = df.replace({np.nan: None})
        
        # Parse dates flexibly
        df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True)
        
        # Calculate daily totals
        daily_totals = df.groupby("Date").agg({
            "Withdrawal": "sum",
            "Deposited": "sum",
            "Balance": "last"
        }).fillna(0)

        return {
            "withdrawal": [
                {"date": date.strftime("%d-%b"), "value": float(row["Withdrawal"])}
                for date, row in daily_totals.iterrows()
            ],
            "deposit": [
                {"date": date.strftime("%d-%b"), "value": float(row["Deposited"])}
                for date, row in daily_totals.iterrows()
            ],
            "balance": [
                {"date": date.strftime("%d-%b"), "value": float(row["Balance"])}
                for date, row in daily_totals.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions")
def get_transactions():
    df = pd.read_csv(WORKING_CSV)
    df = df.replace({np.nan: None})  # Replace NaN with None
    return df[['UPI_Name', 'UPI_Description', 'Date_Formated', 
               'Withdrawal', 'Deposited', 'Balance', 
               'Cumulative_Withdrawal', 'Cumulative_Deposited']].to_dict(orient='records')

@app.get("/upi-analysis")
def get_upi_analysis():
    df = pd.read_csv(WORKING_CSV)
    df = df.replace({np.nan: None})  # Replace NaN with None

    # Convert "Date" column to datetime, handle errors
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True, errors='coerce')

    # Drop rows where Date conversion failed
    df = df.dropna(subset=["Date"])

    # Fill NaN with 0 for calculations
    df["Withdrawal"] = pd.to_numeric(df["Withdrawal"], errors='coerce').fillna(0)

    # Ensure "UPI_Name" column is a string
    df["UPI_Name"] = df["UPI_Name"].astype(str)

    # Group by UPI Name and sum withdrawals
    upi_summary = df.groupby('UPI_Name')['Withdrawal'].sum().sort_values(ascending=False)

    # Get highest daily spend
    daily_spend = df.groupby("Date")["Withdrawal"].sum()

    if not daily_spend.empty:
        highest_spend_date = daily_spend.idxmax()
        highest_spend_amount = daily_spend.max()
    else:
        highest_spend_date = None
        highest_spend_amount = 0.0

    # Find highest individual transaction
    # Check if the DataFrame is empty or if all values are zero
    if not df.empty and df["Withdrawal"].max() > 0:
        max_withdrawal_idx = df["Withdrawal"].idxmax()
        highest_transaction = {
            "amount": float(df.loc[max_withdrawal_idx, "Withdrawal"]),
            "date": df.loc[max_withdrawal_idx, "Date"].strftime("%d %B %Y"),
            "description": df.loc[max_withdrawal_idx, "UPI_Description"]
        }
    else:
        highest_transaction = {"amount": 0.0, "date": None, "description": None}

    return {
        "upiWise": [
            {"name": name, "amount": float(amount)}
            for name, amount in upi_summary.items()
            if name.strip() and name != "nan"  # Skip empty names and 'nan' strings
        ],
        "highestTransaction": highest_transaction,
        "highestDailySpend": {
            "date": highest_spend_date.strftime("%d %B %Y") if highest_spend_date else None,
            "amount": float(highest_spend_amount)
        }
    }

def process_data_for_analysis(csv_path: Path) -> None:
    """Process CSV data and prepare it for analysis"""
    df = pd.read_csv(csv_path)
    # Try to parse dates flexibly
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True)
    df["Date_Formated"] = df["Date"].dt.strftime("%d-%b-%Y")
    df.to_csv(WORKING_CSV, index=False)




model = joblib.load('loan_status_predictor.pkl')


num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
scaler = joblib.load('vector.pkl')

# Gender	Married	Dependents	Education	Self_Employed	ApplicantIncome	CoapplicantIncome	LoanAmount	Loan_Amount_Term	Credit_History	Property_Area

class LoanApproval(BaseModel):
    Gender: float 
    Married: float
    Dependents:float
    Education: float
    Self_Employed: float
    ApplicantIncome:float
    CoapplicantIncome:float
    LoanAmount:float 
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: float

@app.post("/predict")
async def predict_loan_status(application: LoanApproval):
    input_data = pd.DataFrame([application.dict()])
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    result = model.predict(input_data)

    if result[0] == 1:
        return {'Loan Status': "Approved"}
    else:
        return {'Loan Status': "Not Approved"}


@app.get("/install-ghostscript")
def install_ghostscript():
    try:
        os.system("apt-get update && apt-get install -y ghostscript")
        return {"message": "✅ Ghostscript installed successfully!"}
    except Exception as e:
        return {"error": f"❌ Failed to install Ghostscript: {str(e)}"}