/* =====================================================
   SmartSpend AI - Dashboard JavaScript

   The frontend ONLY:
     1. collects the user's input
     2. sends it to FastAPI (POST /predict)
     3. displays the JSON that comes back
   There is NO machine-learning code here. XGBoost lives in the backend.
   ===================================================== */

// ---------- 1. Settings ----------
const API_URL = "https://budget-recommendation-model.onrender.com";

const CONNECTION_ERROR_MESSAGE =
    "Unable to connect to SmartSpend AI. Please make sure the backend server is running.";
const HTTP_ERROR_MESSAGE =
    "Unable to generate your recommendation. Please check your financial information and try again.";

// The nine expense categories (names MUST match the backend exactly)
const EXPENSE_KEYS = [
    "Rent", "Groceries", "Transport", "Eating_Out", "Entertainment",
    "Utilities", "Healthcare", "Education", "Miscellaneous"
];

// Friendly names to show on screen
const CATEGORY_LABELS = {
    Rent: "Rent",
    Groceries: "Groceries",
    Transport: "Transport",
    Eating_Out: "Eating Out",
    Entertainment: "Entertainment",
    Utilities: "Utilities",
    Healthcare: "Healthcare",
    Education: "Education",
    Miscellaneous: "Miscellaneous"
};

// Backend field name -> id of the HTML input
const FIELD_IDS = {
    Income: "income",
    Age: "age",
    Dependents: "dependents",
    Occupation: "occupation",
    City_Tier: "cityTier",
    Rent: "rent",
    Groceries: "groceries",
    Transport: "transport",
    Eating_Out: "eatingOut",
    Entertainment: "entertainment",
    Utilities: "utilities",
    Healthcare: "healthcare",
    Education: "education",
    Miscellaneous: "miscellaneous"
};

// One colour per expense category (used by the chart and the list dots)
const CATEGORY_COLORS = [
    "#8b5cf6", "#34d399", "#60a5fa", "#fbbf24", "#f472b6",
    "#22d3ee", "#fb923c", "#a3e635", "#94a3b8"
];

// Keep the chart objects so we can destroy them before drawing new ones
let expenseChart = null;
let overviewChart = null;

// Short helper: get an element by its id
function byId(id) {
    return document.getElementById(id);
}

// ---------- 2. Formatting helpers ----------

// 50000 -> "₹50,000"
function formatCurrency(value) {
    const number = Number(value);
    return new Intl.NumberFormat("en-IN", {
        style: "currency",
        currency: "INR",
        maximumFractionDigits: 0
    }).format(Number.isFinite(number) ? number : 0);
}

// 46 -> "46.0%"
function formatPercent(value) {
    const number = Number(value);
    return (Number.isFinite(number) ? number : 0).toFixed(1) + "%";
}

// ---------- 3. Reading and validating the form ----------

// Read every input and build the JSON object that FastAPI expects
function getFormData() {
    const data = {};

    for (const [fieldName, inputId] of Object.entries(FIELD_IDS)) {
        const rawValue = byId(inputId).value.trim();

        if (fieldName === "Occupation" || fieldName === "City_Tier") {
            data[fieldName] = rawValue;                      // text values
        } else if (rawValue === "") {
            // An empty expense counts as 0; other empty numbers are invalid (NaN)
            data[fieldName] = EXPENSE_KEYS.includes(fieldName) ? 0 : NaN;
        } else {
            data[fieldName] = Number(rawValue);              // numeric values
        }
    }
    return data;
}

// Check the data and return a list of problems (empty list = everything OK)
function validateForm(data) {
    const errors = [];

    if (!Number.isFinite(data.Income) || data.Income <= 0) {
        errors.push("Monthly income must be greater than 0.");
    }
    if (!Number.isInteger(data.Age) || data.Age < 1 || data.Age > 120) {
        errors.push("Age must be a whole number between 1 and 120.");
    }
    if (!Number.isInteger(data.Dependents) || data.Dependents < 0 || data.Dependents > 20) {
        errors.push("Dependents must be a whole number between 0 and 20.");
    }
    if (data.Occupation === "") {
        errors.push("Please select your occupation.");
    }
    if (data.City_Tier === "") {
        errors.push("Please select your city tier.");
    }
    for (const key of EXPENSE_KEYS) {
        if (!Number.isFinite(data[key]) || data[key] < 0) {
            errors.push(`${CATEGORY_LABELS[key]} must be 0 or more.`);
        }
    }
    return errors;
}

// ---------- 4. Loading and error messages ----------

// Turn the loading state on/off
function showLoading(isLoading) {
    byId("predictBtn").disabled = isLoading;               // stop duplicate clicks
    byId("spinner").hidden = !isLoading;
    byId("btnText").textContent = isLoading ? "Analyzing Finances..." : "Generate Smart Budget";
}

// Show a styled error message under the form (accepts text or a list of texts)
function showError(messages) {
    const errorBox = byId("errorBox");
    const list = Array.isArray(messages) ? messages : [messages];

    errorBox.innerHTML = "";
    const ul = document.createElement("ul");
    for (const message of list) {
        const li = document.createElement("li");
        li.textContent = message;                           // textContent = safe
        ul.appendChild(li);
    }
    errorBox.appendChild(ul);
    errorBox.hidden = false;
}

function clearError() {
    byId("errorBox").hidden = true;
    byId("errorBox").innerHTML = "";
}

// ---------- 5. Talking to FastAPI ----------

// Send the data to POST /predict and return the JSON answer
async function sendPredictionRequest(financialData) {
    let response;

    try {
        response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(financialData)
        });
    } catch (networkError) {
        // fetch() fails like this when the server is off (or blocked by CORS)
        throw new Error(CONNECTION_ERROR_MESSAGE);
    }

    if (!response.ok) {
        const error = new Error(HTTP_ERROR_MESSAGE);
        error.details = await readErrorMessages(response);
        throw error;
    }

    const data = await response.json();
    if (!data || !data.summary || !data.budget) {
        throw new Error(HTTP_ERROR_MESSAGE);
    }
    return data;
}

// Pull a useful message out of an error response from FastAPI
async function readErrorMessages(response) {
    try {
        const body = await response.json();

        // 422 = our custom validation errors: { errors: ["Rent: ...", ...] }
        if (Array.isArray(body.errors) && body.errors.length > 0) {
            return body.errors;
        }
        // 500 / 503 = a short, safe message in "detail"
        if (typeof body.detail === "string" && response.status >= 500) {
            return [body.detail];
        }
        // Default FastAPI validation format
        if (Array.isArray(body.detail)) {
            return body.detail.map(item => item.msg);
        }
    } catch (parseError) {
        // response was not JSON - fall through to the generic message
    }
    return [HTTP_ERROR_MESSAGE];
}

// ---------- 6. Updating the dashboard ----------

// Master function: show the results area, then fill every part
function updateDashboard(data) {
    byId("emptyState").hidden = true;
    byId("results").hidden = false;          // show BEFORE drawing charts (they need a size)

    updateSummaryCards(data.summary, data.budget);
    updateBudgetSection(data.budget);
    updateFinancialHealth(data.financial_health);
    updateExpenseBreakdown(data.category_spending);
    updateCharts(data);
    displayInsights(data.insights);
}

// Four cards at the top
function updateSummaryCards(summary, budget) {
    byId("cardIncome").textContent = formatCurrency(summary.total_income);
    byId("cardExpenses").textContent = formatCurrency(summary.total_expenses);
    byId("cardBudget").textContent = formatCurrency(budget.recommended_budget);

    const savingsElement = byId("cardSavings");
    savingsElement.textContent = formatCurrency(summary.savings);
    savingsElement.className = "stat-value " + (summary.savings >= 0 ? "positive" : "negative");

    byId("cardSavingsPercent").textContent =
        formatPercent(summary.savings_percentage) + " of your income";
}

// Recommended vs actual spending + progress bar
function updateBudgetSection(budget) {
    const difference = Number(budget.difference);
    const percentUsed = Number(budget.percentage_used);

    byId("budgetRecommended").textContent = formatCurrency(budget.recommended_budget);
    byId("budgetActual").textContent = formatCurrency(budget.actual_spending);
    byId("budgetPercent").textContent = formatPercent(percentUsed);

    // Positive difference = money left, negative = overspent
    byId("budgetDifferenceLabel").textContent = difference >= 0 ? "Remaining" : "Over Budget By";
    const differenceElement = byId("budgetDifference");
    differenceElement.textContent = formatCurrency(Math.abs(difference));
    differenceElement.className = "mid-number " + (difference >= 0 ? "positive" : "negative");

    // Choose colours from the status the backend returned
    let badgeClass = "badge-good";
    let fillClass = "";
    if (budget.status === "Over Budget") {
        badgeClass = "badge-bad";
        fillClass = "fill-bad";
    } else if (budget.status === "Budget Fully Used") {
        badgeClass = "badge-caution";
        fillClass = "fill-caution";
    }

    const badge = byId("budgetStatus");
    badge.textContent = budget.status;
    badge.className = "status-badge " + badgeClass;

    // Progress bar (never wider than 100%)
    byId("progressText").textContent =
        `${formatCurrency(budget.actual_spending)} / ${formatCurrency(budget.recommended_budget)}`;
    byId("progressPercentText").textContent = formatPercent(percentUsed) + " used";

    const fill = byId("progressFill");
    fill.className = "progress-fill " + fillClass;
    fill.style.width = "0%";                                  // restart the animation
    requestAnimationFrame(() => {
        fill.style.width = Math.min(Math.max(percentUsed, 0), 100) + "%";
    });
}

// Circular score
function updateFinancialHealth(health) {
    const score = Math.max(0, Math.min(100, Number(health.score) || 0));

    // Use the status from FastAPI; only calculate it ourselves if it is missing
    let status = health.status;
    if (!status) {
        if (score >= 80) status = "Excellent";
        else if (score >= 60) status = "Good";
        else if (score >= 40) status = "Needs Improvement";
        else status = "Poor";
    }

    const looks = {
        "Excellent": { color: "var(--green)",  text: "Your spending and savings habits are currently healthy." },
        "Good": { color: "var(--blue)",        text: "You are doing well, with a little room to improve." },
        "Needs Improvement": { color: "var(--yellow)", text: "Some spending habits need attention to protect your savings." },
        "Poor": { color: "var(--red)",         text: "Your spending is putting pressure on your finances." }
    };
    const look = looks[status] || looks["Good"];

    const ring = byId("healthRing");
    ring.style.setProperty("--score", score);
    ring.style.setProperty("--ring-color", look.color);

    byId("healthScore").textContent = score;
    const statusElement = byId("healthStatus");
    statusElement.textContent = status;
    statusElement.style.color = look.color;
    byId("healthDescription").textContent = look.text;
}

// List of the nine categories with amount and % of total expenses
function updateExpenseBreakdown(categorySpending) {
    const list = byId("breakdownList");
    list.innerHTML = "";

    const total = EXPENSE_KEYS.reduce((sum, key) => sum + (Number(categorySpending[key]) || 0), 0);

    EXPENSE_KEYS.forEach((key, index) => {
        const amount = Number(categorySpending[key]) || 0;
        const percent = total > 0 ? (amount / total) * 100 : 0;    // calculated dynamically

        const row = document.createElement("div");
        row.className = "breakdown-row";

        const dot = document.createElement("span");
        dot.className = "dot";
        dot.style.background = CATEGORY_COLORS[index];

        const name = document.createElement("span");
        name.textContent = CATEGORY_LABELS[key];

        const amountElement = document.createElement("span");
        amountElement.className = "breakdown-amount";
        amountElement.textContent = formatCurrency(amount);

        const percentElement = document.createElement("span");
        percentElement.className = "breakdown-percent";
        percentElement.textContent = Math.round(percent) + "%";

        row.append(dot, name, amountElement, percentElement);
        list.appendChild(row);
    });
}

// Make Chart.js match the dark theme
function configureChartTheme() {
    Chart.defaults.color = "#8b95a8";
    Chart.defaults.borderColor = "#232c3f";
    Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
}

// Doughnut chart + bar chart (old charts are destroyed first)
function updateCharts(data) {
    if (typeof Chart === "undefined") {
        // Chart.js could not be loaded (for example, no internet connection)
        document.querySelectorAll(".chart-box").forEach(box => {
            box.innerHTML = '<p class="chart-note">Charts could not be loaded. Check your internet connection.</p>';
        });
        return;
    }
    configureChartTheme();

    // --- Doughnut: expense categories (from API category_spending) ---
    if (expenseChart) expenseChart.destroy();
    const spending = data.category_spending;
    const categoryAmounts = EXPENSE_KEYS.map(key => Number(spending[key]) || 0);
    const categoryTotal = categoryAmounts.reduce((a, b) => a + b, 0);

    expenseChart = new Chart(byId("expenseChart"), {
        type: "doughnut",
        data: {
            labels: EXPENSE_KEYS.map(key => CATEGORY_LABELS[key]),
            datasets: [{
                data: categoryAmounts,
                backgroundColor: CATEGORY_COLORS,
                borderColor: "#121826",
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "65%",
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: context => {
                            const percent = categoryTotal > 0 ? (context.parsed / categoryTotal) * 100 : 0;
                            return ` ${context.label}: ${formatCurrency(context.parsed)} (${percent.toFixed(0)}%)`;
                        }
                    }
                }
            }
        }
    });

    // --- Bar chart: income, expenses, savings, recommended budget ---
    if (overviewChart) overviewChart.destroy();
    const summary = data.summary;
    const savingsColor = summary.savings >= 0 ? "#34d399" : "#f87171";

    overviewChart = new Chart(byId("overviewChart"), {
        type: "bar",
        data: {
            labels: ["Income", "Expenses", "Savings", "Recommended Budget"],
            datasets: [{
                data: [
                    summary.total_income,
                    summary.total_expenses,
                    summary.savings,
                    data.budget.recommended_budget
                ],
                backgroundColor: ["#60a5fa", "#fb923c", savingsColor, "#8b5cf6"],
                borderRadius: 8,
                maxBarThickness: 70
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: context => " " + formatCurrency(context.parsed.y) } }
            },
            scales: {
                y: {
                    ticks: {
                        callback: value => "₹" + new Intl.NumberFormat("en-IN", { notation: "compact" }).format(value)
                    }
                },
                x: { grid: { display: false } }
            }
        }
    });
}

// Show the insights array from FastAPI as small cards
function displayInsights(insights) {
    const container = byId("insightsList");
    container.innerHTML = "";

    if (!Array.isArray(insights) || insights.length === 0) {
        container.textContent = "No insights available.";
        return;
    }

    for (const text of insights) {
        const lower = text.toLowerCase();

        // Pick an icon/colour only for DISPLAY, based on the wording of the insight
        let icon = "📊";
        let tone = "";
        if (lower.includes("above") || lower.includes("overspending") || lower.includes("over your")) {
            icon = "⚠️";
            tone = "tone-warning";
        } else if (lower.includes("saving")) {
            icon = "💡";
            tone = "tone-good";
        } else if (lower.includes("remaining") || lower.includes("within")) {
            icon = "✓";
            tone = "tone-good";
        }

        const card = document.createElement("div");
        card.className = "insight " + tone;

        const iconElement = document.createElement("span");
        iconElement.className = "insight-icon";
        iconElement.textContent = icon;

        const textElement = document.createElement("span");
        textElement.className = "insight-text";
        textElement.textContent = text;                  // text comes from the API

        card.append(iconElement, textElement);
        container.appendChild(card);
    }
}

// ---------- 7. Form submit handler ----------

async function handleSubmit(event) {
    event.preventDefault();                  // stop the page from reloading
    clearError();

    const financialData = getFormData();
    const problems = validateForm(financialData);
    if (problems.length > 0) {
        showError(problems);
        return;
    }

    showLoading(true);
    try {
        const data = await sendPredictionRequest(financialData);
        updateDashboard(data);
        byId("results").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        showError(error.details || error.message);
    } finally {
        showLoading(false);                  // always restore the button
    }
}

// Fill the form with example INPUT values (not results) to make testing quick
function fillSampleValues() {
    const sample = {
        Income: 50000, Age: 24, Dependents: 1,
        Occupation: "Salaried", City_Tier: "Tier_2",
        Rent: 10000, Groceries: 5000, Transport: 2500,
        Eating_Out: 2000, Entertainment: 1500, Utilities: 2500,
        Healthcare: 1000, Education: 1000, Miscellaneous: 1500
    };
    for (const [fieldName, inputId] of Object.entries(FIELD_IDS)) {
        byId(inputId).value = sample[fieldName];
    }
    clearError();
}

// ---------- 8. Sidebar behaviour ----------

function setActiveNav(sectionId) {
    document.querySelectorAll(".nav-link").forEach(link => {
        link.classList.toggle("active", link.dataset.target === sectionId);
    });
}

function openSidebar(isOpen) {
    byId("sidebar").classList.toggle("open", isOpen);
    byId("overlay").classList.toggle("show", isOpen);
}

function setupSidebar() {
    // Click a menu item -> smooth scroll to its section
    document.querySelectorAll(".nav-link").forEach(link => {
        link.addEventListener("click", event => {
            event.preventDefault();
            let target = byId(link.dataset.target);

            // Result sections are hidden until a prediction exists -> go to the empty state
            if (!target || target.offsetParent === null) {
                target = byId("emptyState");
            }
            target.scrollIntoView({ behavior: "smooth", block: "start" });
            setActiveNav(link.dataset.target);
            openSidebar(false);              // close the menu on mobile
        });
    });

    byId("menuBtn").addEventListener("click", () => openSidebar(true));
    byId("overlay").addEventListener("click", () => openSidebar(false));

    // Highlight the menu item of the section currently on screen
    window.addEventListener("scroll", () => {
        let current = "section-dashboard";
        document.querySelectorAll(".nav-link").forEach(link => {
            const section = byId(link.dataset.target);
            if (section && section.offsetParent !== null && section.getBoundingClientRect().top <= 140) {
                current = link.dataset.target;
            }
        });
        setActiveNav(current);
    });
}

// ---------- 9. Start the app ----------
byId("financialForm").addEventListener("submit", handleSubmit);
byId("sampleBtn").addEventListener("click", fillSampleValues);
setupSidebar();
