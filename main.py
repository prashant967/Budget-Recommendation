"""
SmartSpend AI - FastAPI Backend

This backend ONLY loads the already-trained pipeline (smartspend_budget_model.pkl)
and uses it for prediction. It never trains or fits anything.

Run with:  uvicorn main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

import logging
import re
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("smartspend")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# 1. Constants (must match the notebook used for training)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "smartspend_budget_model.pkl"

EXPENSE_CATEGORIES = [
    "Rent",
    "Groceries",
    "Transport",
    "Eating_Out",
    "Entertainment",
    "Utilities",
    "Healthcare",
    "Education",
    "Miscellaneous",
]

# Exactly the columns (and order) used when the model was trained
FEATURE_COLUMNS = [
    "Income",
    "Age",
    "Dependents",
    "Occupation",
    "City_Tier",
] + EXPENSE_CATEGORIES

# Spending that is "nice to have" rather than essential
DISCRETIONARY_CATEGORIES = ["Eating_Out", "Entertainment", "Miscellaneous"]

# ---------------------------------------------------------------------------
# 2. Load the saved ML pipeline ONCE, when the server starts
# ---------------------------------------------------------------------------

model = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info("SmartSpend model loaded successfully.")
except Exception as error:
    # The server still starts, but prediction endpoints will return a clear error
    logger.error("Unable to load SmartSpend model. Details: %s", error)

# ---------------------------------------------------------------------------
# 3. Create the FastAPI app + CORS (so React can call this API)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SmartSpend AI API",
    description="AI-powered budget recommendation API",
    version="1.0.0",
)

# localhost:3000 -> Create React App | localhost:5173 -> Vite
# Only these development origins are allowed (not "*").
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",

    # deployed frontend (React) on Render.com
    "https://budget-recommendation-ui.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 4. Request model (Pydantic validates every incoming request)
# ---------------------------------------------------------------------------


class FinancialInput(BaseModel):
    Income: float = Field(..., gt=0, description="Monthly income in rupees (> 0)")
    Age: int = Field(..., gt=0, le=120, description="Age in years (> 0)")
    Dependents: int = Field(..., ge=0, le=20, description="Number of dependents (>= 0)")
    Occupation: Literal["Student", "Salaried", "Self-Employed", "Freelancer", "Business", "Other"]
    City_Tier: Literal["Tier_1", "Tier_2", "Tier_3"]

    Rent: float = Field(..., ge=0)
    Groceries: float = Field(..., ge=0)
    Transport: float = Field(..., ge=0)
    Eating_Out: float = Field(..., ge=0)
    Entertainment: float = Field(..., ge=0)
    Utilities: float = Field(..., ge=0)
    Healthcare: float = Field(..., ge=0)
    Education: float = Field(..., ge=0)
    Miscellaneous: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Income": 50000,
                "Age": 24,
                "Dependents": 1,
                "Occupation": "Salaried",
                "City_Tier": "Tier_2",
                "Rent": 10000,
                "Groceries": 5000,
                "Transport": 2500,
                "Eating_Out": 2000,
                "Entertainment": 1500,
                "Utilities": 2500,
                "Healthcare": 1000,
                "Education": 1000,
                "Miscellaneous": 1500,
            }
        }
    }


# Turn Pydantic's technical errors into short, readable messages
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    problems = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"] if part != "body")
        problems.append(f"{field}: {error['msg']}")
    return JSONResponse(
        status_code=422,
        content={"success": False, "message": "Invalid input data.", "errors": problems},
    )


# ---------------------------------------------------------------------------
# 5. Helper functions
# ---------------------------------------------------------------------------


def format_rupees(amount):
    """Format a number in Indian style, e.g. 1234567 -> ₹12,34,567"""
    amount = int(round(amount))
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))
    if len(digits) > 3:
        last_three = digits[-3:]
        rest = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", digits[:-3])
        digits = rest + "," + last_three
    return f"{sign}₹{digits}"


def calculate_total_expenses(user_data):
    """Add up all 9 expense categories."""
    return float(sum(getattr(user_data, category) for category in EXPENSE_CATEGORIES))


def calculate_savings(income, total_expenses):
    """Savings = Income - Expenses. Savings % = Savings / Income x 100."""
    if income <= 0:
        return 0.0, 0.0  # safe fallback for invalid income
    savings = income - total_expenses
    savings_percentage = (savings / income) * 100
    return float(savings), float(savings_percentage)


def predict_budget(user_data):
    """Use the saved ML pipeline to predict the recommended monthly budget."""
    # One-row DataFrame with exactly the same columns the model was trained on
    user_df = pd.DataFrame([user_data.model_dump()])[FEATURE_COLUMNS]

    # The pipeline already contains OneHotEncoder + XGBoost -> only predict()
    prediction = float(model.predict(user_df)[0])

    if not np.isfinite(prediction):
        raise ValueError("Model returned an invalid prediction.")

    prediction = max(0.0, prediction)                    # never negative
    prediction = round(prediction / 100) * 100           # user-friendly (nearest Rs.100)
    prediction = min(prediction, float(user_data.Income))  # never above income
    return float(prediction)


def calculate_budget_status(total_expenses, recommended_budget):
    """Compare actual spending with the recommended budget."""
    difference = recommended_budget - total_expenses
    if recommended_budget > 0:
        percentage_used = (total_expenses / recommended_budget) * 100
    else:
        percentage_used = 0.0  # avoid division by zero

    if round(difference, 2) == 0:
        status = "Budget Fully Used"
    elif difference > 0:
        status = "Within Budget"
    else:
        status = "Over Budget"

    return {
        "recommended_budget": round(recommended_budget, 2),
        "actual_spending": round(total_expenses, 2),
        "difference": round(difference, 2),
        "percentage_used": round(percentage_used, 1),
        "status": status,
    }


def calculate_financial_health(income, total_expenses, recommended_budget, category_spending):
    """
    Rule-based score out of 100 (NOT predicted by XGBoost):
      - Savings rate .............. 35 points (full marks at 20%+ savings)
      - Expense-to-income ratio ... 25 points (full at <= 70%, zero at 100%+)
      - Budget adherence .......... 25 points (full if within budget, zero if 25%+ over)
      - Discretionary spending .... 15 points (full at <= 10% of income, zero at 25%+)
        (Eating_Out + Entertainment + Miscellaneous)
    """
    if income <= 0:
        return {"score": 0, "status": "Poor"}

    # 1. Savings rate
    savings_percentage = (income - total_expenses) / income * 100
    savings_points = 35 * np.clip(savings_percentage / 20, 0, 1)

    # 2. Expense-to-income ratio
    expense_ratio = total_expenses / income
    ratio_points = 25 * np.clip((1.0 - expense_ratio) / 0.30, 0, 1)

    # 3. Budget adherence
    if recommended_budget <= 0:
        budget_points = 0.0
    elif total_expenses <= recommended_budget:
        budget_points = 25.0
    else:
        overshoot = (total_expenses - recommended_budget) / recommended_budget
        budget_points = 25 * np.clip(1 - overshoot / 0.25, 0, 1)

    # 4. Discretionary spending
    discretionary_total = sum(category_spending[c] for c in DISCRETIONARY_CATEGORIES)
    discretionary_ratio = discretionary_total / income
    discretionary_points = 15 * np.clip((0.25 - discretionary_ratio) / 0.15, 0, 1)

    score = int(round(savings_points + ratio_points + budget_points + discretionary_points))
    score = max(0, min(100, score))

    if score >= 80:
        status = "Excellent"
    elif score >= 60:
        status = "Good"
    elif score >= 40:
        status = "Needs Improvement"
    else:
        status = "Poor"

    return {"score": score, "status": status}


def generate_insights(user_data, total_expenses, savings, savings_percentage, recommended_budget):
    """Create 5 plain-English insights. Every number is calculated from the input."""
    income = user_data.Income
    insights = []

    # 1. Savings
    if savings >= 0:
        insights.append(
            f"You are saving {savings_percentage:.1f}% of your monthly income ({format_rupees(savings)})."
        )
    else:
        insights.append(
            f"You are overspending by {format_rupees(-savings)}, "
            f"which is {-savings_percentage:.1f}% more than your income."
        )

    # 2. Rent
    rent_percentage = user_data.Rent / income * 100
    rent_note = " This is above the commonly suggested 30%." if rent_percentage > 30 else ""
    insights.append(f"Rent accounts for {rent_percentage:.1f}% of your income.{rent_note}")

    # 3. Eating out + entertainment
    fun_spending = user_data.Eating_Out + user_data.Entertainment
    fun_percentage = fun_spending / income * 100
    insights.append(
        f"Eating out and entertainment account for {fun_percentage:.1f}% of your income "
        f"({format_rupees(fun_spending)})."
    )

    # 4. Compare with recommended budget
    difference = recommended_budget - total_expenses
    if round(difference, 2) == 0:
        insights.append("You have used exactly your recommended monthly budget.")
    elif difference > 0:
        insights.append(
            f"You have {format_rupees(difference)} remaining within your recommended monthly budget."
        )
    else:
        insights.append(
            f"You are {format_rupees(-difference)} above your recommended monthly budget."
        )

    # 5. Largest variable expense (everything except Rent)
    variable_expenses = {c: getattr(user_data, c) for c in EXPENSE_CATEGORIES if c != "Rent"}
    largest_name = max(variable_expenses, key=variable_expenses.get)
    largest_amount = variable_expenses[largest_name]
    insights.append(
        f"{largest_name.replace('_', ' ')} is your largest variable expense "
        f"({format_rupees(largest_amount)}, {largest_amount / income * 100:.1f}% of income)."
    )

    return insights


def create_friendly_output(result):
    """Turn the result dictionary into a readable text report."""
    summary = result["summary"]
    budget = result["budget"]
    health = result["financial_health"]

    # Show remaining budget, or the overspend amount if over budget
    if budget["difference"] >= 0:
        remaining_label, remaining_value = "Budget Remaining:", budget["difference"]
    else:
        remaining_label, remaining_value = "Over Budget By:", -budget["difference"]

    lines = [
        "SMARTSPEND AI",
        "",
        f"{'Monthly Income:':<29}{format_rupees(summary['total_income'])}",
        f"{'Monthly Expenses:':<29}{format_rupees(summary['total_expenses'])}",
        f"{'Monthly Savings:':<29}{format_rupees(summary['savings'])}",
        f"{'Savings Percentage:':<29}{summary['savings_percentage']:.1f}%",
        "",
        f"{'Recommended Monthly Budget:':<29}{format_rupees(budget['recommended_budget'])}",
        f"{'Actual Spending:':<29}{format_rupees(budget['actual_spending'])}",
        f"{remaining_label:<29}{format_rupees(remaining_value)}",
        f"{'Budget Status:':<29}{budget['status']}",
        "",
        f"{'Financial Health Score:':<29}{health['score']}/100",
        f"{'Financial Health:':<29}{health['status']}",
        "",
        "Smart Insights:",
        "",
    ]
    lines += [f"• {insight}" for insight in result["insights"]]
    return "\n".join(lines)


def build_result(user_data):
    """Run the full flow and return the dashboard dictionary."""
    # Calculate spending numbers
    total_expenses = calculate_total_expenses(user_data)
    savings, savings_percentage = calculate_savings(user_data.Income, total_expenses)

    # Saved ML pipeline -> recommended budget (with a safe error if it fails)
    try:
        recommended_budget = predict_budget(user_data)
    except Exception as error:
        logger.error("Prediction failed: %s", error)  # details stay in the server log
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.")

    category_spending = {c: getattr(user_data, c) for c in EXPENSE_CATEGORIES}

    budget = calculate_budget_status(total_expenses, recommended_budget)
    financial_health = calculate_financial_health(
        user_data.Income, total_expenses, recommended_budget, category_spending
    )
    insights = generate_insights(
        user_data, total_expenses, savings, savings_percentage, recommended_budget
    )

    return {
        "success": True,
        "summary": {
            "total_income": user_data.Income,
            "total_expenses": round(total_expenses, 2),
            "savings": round(savings, 2),
            "savings_percentage": round(savings_percentage, 1),
        },
        "category_spending": category_spending,
        "budget": budget,
        "financial_health": financial_health,
        "insights": insights,
    }


def require_model():
    """Stop the request with a clear message if the model is not loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Unable to load SmartSpend model.")


# ---------------------------------------------------------------------------
# 6. Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {"message": "SmartSpend Budget Recommendation API is running", "status": "healthy"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model": "XGBoost Budget Recommendation",
    }


@app.post("/predict")
def predict(user_data: FinancialInput, friendly: bool = False):
    """Main endpoint: returns dashboard JSON. Add ?friendly=true for a text report too."""
    require_model()
    result = build_result(user_data)
    if friendly:
        result["friendly_output"] = create_friendly_output(result)
    return result


@app.post("/predict/summary")
def predict_summary(user_data: FinancialInput):
    """Short response for quickly showing the key numbers."""
    require_model()
    result = build_result(user_data)

    status = result["budget"]["status"]
    if status == "Within Budget":
        message = "You are currently within your recommended monthly budget."
    elif status == "Budget Fully Used":
        message = "You have used your full recommended monthly budget."
    else:
        message = "You are currently over your recommended monthly budget."

    return {
        "recommended_budget": result["budget"]["recommended_budget"],
        "total_expenses": result["summary"]["total_expenses"],
        "savings": result["summary"]["savings"],
        "financial_health_score": result["financial_health"]["score"],
        "financial_health_status": result["financial_health"]["status"],
        "budget_status": status,
        "message": message,
    }
