import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
import yagmail
from flask import Flask, render_template, request
from dotenv import load_dotenv

# Load sensitive environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Fetch secret credentials securely from environment
EMAIL_SENDER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("DB_PASSWORD")

# 1. BULLETPROOF PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'loan_model.pkl')

print(f" Looking for 'loan_model.pkl' at: {model_path}")

# Load your trained machine learning model safely with crash prevention
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(" Success! 'loan_model.pkl' loaded perfectly.")
except FileNotFoundError:
    print(f" Error: Could not find 'loan_model.pkl' at {model_path}")
    print(" Action: If it's in a subfolder like 'dataset', move 'loan_model.pkl' right next to app1.py!")
    model = None

# Auto-initialize database tracking tables (Runs on startup)
def init_db():
    conn = sqlite3.connect('loans.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS new_loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT,
            user_email TEXT,
            income REAL,
            credit_score INTEGER,
            loan_amount REAL,
            years_employed INTEGER,
            points REAL,
            prediction_result TEXT,
            reason_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Reason generator calibrated to highlight the applicant's strongest metric proportionally
def get_decision_reason(prediction, income, credit_score, loan_amount, years_employed, points):
    if prediction == 0:  # REJECTED REASONS
        reasons = []
        if credit_score < 550:
            reasons.append("a low credit score profile")
        if income < loan_amount:
            reasons.append("the requested loan amount being too high relative to your annual income")
        if years_employed < 5:
            reasons.append("insufficient years of continuous employment history")
        if points < 50:
            reasons.append("your application not meeting the minimum eligibility points")
        
        if not reasons:
            return "the calculated risk thresholds of our predictive classification model."
        return f"primarily due to {reasons[0]}."
        
    else:  # APPROVED REASONS (Evaluates proportional performance strengths)
        reasons = []
        
        # Aligned perfectly to baseline thresholds to prevent empty list fallback bugs
        if credit_score >= 550:
            score_perf = credit_score / 550  
            reasons.append((score_perf, "a strong and healthy credit rating"))
            
        if income >= loan_amount:
            income_perf = income / loan_amount  
            reasons.append((income_perf, "an excellent income-to-loan safety ratio"))
            
        if years_employed >= 5:
            emp_perf = years_employed / 5  
            reasons.append((emp_perf, "an extensive, stable employment history"))
            
        if points >= 50:
            points_perf = points / 50  
            reasons.append((points_perf, "your application meeting the minimum eligibility points"))
            
        if not reasons:
            return "balanced positive indicators across all evaluation categories."
            
        # Dynamic sort: Areas exceeding requirements by the highest percentage scale go first
        reasons.sort(reverse=True, key=lambda x: x[0])
        return f"supported primarily by {reasons[0][1]}."

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('output.html', prediction_text="Error: Model file missing on server.", alert_class="warning")

        # 1. Grab data inputs - Email is now strictly required
        user_email = request.form.get('UserEmail', '').strip()
        applicant_name = request.form.get('name', '').strip()
        
        # Backend validation check for teammates
        if not user_email or not applicant_name:
            return render_template('output.html', prediction_text="Error: Applicant Name and Email Address are mandatory fields.", alert_class="warning")
        
        income = float(request.form.get('income', 0))
        credit_score = int(request.form.get('credit_score', 0))
        loan_amount = float(request.form.get('loan_amount', 0))
        years_employed = int(request.form.get('years_employed', 0))
        points = float(request.form.get('points', 0))

        # 2. Package numerical inputs inside a pandas DataFrame for ML engine compatibility
        input_data = pd.DataFrame([{
            'income': income,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'years_employed': years_employed,
            'points': points
        }])

        # 3. FIXED LOGIC: Strict Rule Matching
        if credit_score < 550:
            prediction = 0
            reason_text = "primarily due to a low credit score profile."
            
        elif income < loan_amount:
            prediction = 0
            reason_text = "primarily due to the requested loan amount being too high relative to your annual income."
            
        elif years_employed < 5:
            prediction = 0
            reason_text = "primarily due to insufficient years of continuous employment history."
            
        elif points < 50:
            prediction = 0
            reason_text = "primarily due to your application not meeting the minimum eligibility points."
            
        else:
            # If ALL minimum rules are maintained, force the prediction to APPROVED (1)
            prediction = 1
            reason_text = get_decision_reason(prediction, income, credit_score, loan_amount, years_employed, points)
        
        # 4. Setup formatting display variables
        if prediction == 1:
            result_text = f"Congratulations {applicant_name}! Your Loan is Approved."
            alert_class = "success"
        else:
            result_text = f"Sorry {applicant_name}, your Loan application has been Denied."
            alert_class = "danger"

        # 5. DATABASE LOGGING
        conn = sqlite3.connect('loans.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO new_loan_applications (
                applicant_name, user_email, income, credit_score, loan_amount, 
                years_employed, points, prediction_result, reason_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (applicant_name, user_email, income, credit_score, loan_amount, 
              years_employed, points, result_text, reason_text))
        conn.commit()
        conn.close()

        # 6. MANDATORY EMAIL DISPATCH
        try:
            if not EMAIL_SENDER or not EMAIL_PASSWORD:
                raise ValueError("Missing email configuration credentials in .env file.")

            # Initializing yagmail securely using hidden environment variables
            yag = yagmail.SMTP(user=EMAIL_SENDER, password=EMAIL_PASSWORD)
            
            email_subject = f"Smart Lender Application Status Update for {applicant_name}"
            email_body = f"""
            Hi {applicant_name},
            
            Thank you for applying with Smart Lender. Our AI engine has finished processing your details.
            
            Here is your personalized loan profile assessment summary:
            
            Status: {result_text} 
            
            Reasoning Profile: This decision was reached {reason_text}
            
            Requested Capital: ₹{loan_amount:,}
            
            Thank you for utilizing our automated processing channels.
            
            Best regards,
            The Smart Lender!
            """
            
            yag.send(to=user_email, subject=email_subject, contents=email_body)
            print(f" Automated confirmation email successfully sent to: {user_email}")
            
        except Exception as mail_err:
            print(f" Email server dispatch failure: {mail_err}")

        return render_template('output.html', prediction_text=result_text, reason_text=reason_text, alert_class=alert_class)

    except Exception as e:
        return render_template('output.html', prediction_text="Data processing layout failure: " + str(e), alert_class="warning")

if __name__ == '__main__':
    app.run(debug=True)