"""
Electric Vehicle Adoption in Bhutan
A beginner Streamlit + scikit-learn project.

IMPORTANT: this app uses SYNTHETIC (made-up) data only.
No real household in Bhutan is represented here.
"""

import base64
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------
# Colour palette
# ----------------------------------------------------------------------

INK = "#0B0C10"      # page background, near black
SLATE = "#1F2833"    # cards and panels
LIGHT = "#C5C6C7"    # body text
CYAN = "#66FCF1"     # highlight colour
TEAL = "#45A29E"     # secondary highlight
CORAL = "#FF6B6B"    # used only for negative model weights

# ----------------------------------------------------------------------
# Page settings
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="EV Adoption in Bhutan",
    page_icon="🔌",
    layout="wide",
)

# ----------------------------------------------------------------------
# Background artwork
#
# The car photo lives in assets/ev-car.png. CSS cannot point at a local file
# in a Streamlit app, so the image is read once, turned into base64 text and
# written straight into the stylesheet below. The CSS then blurs it and fades
# it back, so it reads as a soft shape behind the page rather than a picture.
# ----------------------------------------------------------------------

CAR_IMAGE_PATH = Path(__file__).parent / "assets" / "ev-car.png"


@st.cache_data
def load_background_image(path):
    """Read the car image and return it as a data URI that CSS can use."""
    try:
        encoded = base64.b64encode(Path(path).read_bytes()).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        # If the image is missing the app still runs, just without the artwork.
        return ""


CAR_IMAGE = load_background_image(CAR_IMAGE_PATH)

# ----------------------------------------------------------------------
# Styling and animation
#
# Streamlit has no setting for most of this, so a block of CSS is injected
# into the page. Every rule is commented below. If a future version of
# Streamlit renames an element, that rule is simply ignored and the app
# still works, so none of this can break the deployment.
# ----------------------------------------------------------------------

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Anton&family=Inter:wght@400;600;700&display=swap');

    /* ---- Page background -------------------------------------------- */
    /* Two soft pools of colour over the near-black base. */
    .stApp {{
        background:
            radial-gradient(900px 500px at 12% 8%, rgba(69,162,158,0.20), transparent 60%),
            radial-gradient(700px 500px at 88% 0%, rgba(102,252,241,0.10), transparent 55%),
            {INK};
    }}

    /* The blurred car sits in its own layer behind everything else. */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("{CAR_IMAGE}");
        background-repeat: no-repeat;
        background-position: center 58%;
        background-size: min(1250px, 96%);
        filter: blur(14px) saturate(1.15) brightness(1.18);
        opacity: 0.42;
        pointer-events: none;   /* clicks pass straight through it */
        z-index: 0;
    }}

    /* A dark veil over the car so headings and body text stay readable.
       It is lighter through the middle, which lets the car show through. */
    .stApp::after {{
        content: "";
        position: fixed;
        inset: 0;
        background: linear-gradient(180deg,
            rgba(11,12,16,0.80) 0%,
            rgba(11,12,16,0.40) 50%,
            rgba(11,12,16,0.90) 100%);
        pointer-events: none;
        z-index: 0;
    }}

    /* Keep the real content above the artwork. */
    .block-container {{
        position: relative;
        z-index: 1;
        padding-top: 2rem;
        font-family: 'Inter', sans-serif;
    }}

    /* ---- Hero banner ------------------------------------------------- */
    .hero {{ padding: 2.2rem 0 1.2rem 0; }}
    .hero-kicker {{
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 3px;
        color: {TEAL};
        margin-bottom: 0.6rem;
    }}
    .hero-title {{
        font-family: 'Anton', Impact, sans-serif;
        font-size: clamp(3.2rem, 11vw, 8.5rem);
        line-height: 0.87;
        color: {CYAN};
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 0 60px rgba(102,252,241,0.28);
    }}
    .hero-sub {{
        max-width: 620px;
        margin-top: 1.4rem;
        color: {LIGHT};
        font-size: 1.02rem;
        line-height: 1.65;
    }}
    .hero-rule {{
        width: 90px;
        height: 3px;
        background: {CYAN};
        margin: 1.6rem 0 0.4rem 0;
    }}

    /* ---- Fade in ----------------------------------------------------- */
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to   {{ opacity: 1; transform: none; }}
    }}
    /* Streamlit keeps all four tabs in the page and hides the inactive
       ones, so this animation replays each time a panel is shown again. */
    .stTabs [data-baseweb="tab-panel"] {{ animation: fadeUp 0.4s ease both; }}
    .hero {{ animation: fadeUp 0.7s ease both; }}

    /* ---- Tabs -------------------------------------------------------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid rgba(197,198,199,0.14);
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        padding: 12px 20px;
        color: {LIGHT};
    }}
    .stTabs [aria-selected="true"] {{ color: {CYAN} !important; }}

    /* ---- Number cards ------------------------------------------------ */
    div[data-testid="stMetric"] {{
        background: rgba(31,40,51,0.72);
        border: 1px solid rgba(102,252,241,0.16);
        border-radius: 12px;
        padding: 16px 18px;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-4px);
        border-color: {CYAN};
        box-shadow: 0 10px 28px rgba(102,252,241,0.14);
    }}
    div[data-testid="stMetricValue"] {{ color: {CYAN}; }}

    /* ---- Bordered containers ----------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 12px;
    }}

    /* ---- Headings and text ------------------------------------------- */
    h1, h2, h3 {{ color: #FFFFFF; letter-spacing: -0.4px; }}
    h2, h3 {{ margin-top: 0.4rem; }}
    a {{ color: {CYAN} !important; }}

    /* Hide the default Streamlit footer for a cleaner page. */
    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Chart styling
#
# The charts are drawn by matplotlib, which defaults to a white background.
# On a dark page that would show up as white boxes, so the figures are made
# transparent and the text and gridlines are switched to the palette.
# ----------------------------------------------------------------------

plt.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": LIGHT,
        "axes.labelcolor": LIGHT,
        "axes.titlecolor": "#FFFFFF",
        "axes.edgecolor": "#2A3642",
        "xtick.color": LIGHT,
        "ytick.color": LIGHT,
        "grid.color": "#1A232D",
        "axes.grid": True,
        "font.size": 10,
    }
)

RANDOM_SEED = 42
N_RECORDS = 800
TARGET_NAME = "ev_adoption"
POSITIVE_LABEL = "Likely to adopt"
NEGATIVE_LABEL = "Not likely yet"

DZONGKHAGS = [
    "Thimphu",
    "Paro",
    "Chhukha",
    "Punakha",
    "Wangdue Phodrang",
    "Bumthang",
    "Trashigang",
    "Samtse",
]
AREA_TYPES = ["Urban", "Semi-urban", "Rural"]
INCOME_BANDS = [
    "Below Nu 20,000",
    "Nu 20,000 - 40,000",
    "Nu 40,000 - 70,000",
    "Above Nu 70,000",
]
CHARGING_OPTIONS = [
    "Home charging possible",
    "Public station nearby",
    "No reliable access",
]
TERRAIN_TYPES = ["Flat valley", "Hilly", "Steep mountain"]
YES_NO = ["Yes", "No"]

NUMERIC_FEATURES = [
    "age",
    "daily_commute_km",
    "current_vehicle_age_years",
    "monthly_fuel_cost_nu",
    "electricity_hours_per_day",
]
CATEGORICAL_FEATURES = [
    "dzongkhag",
    "area_type",
    "income_band",
    "charging_access",
    "terrain",
    "knows_about_subsidy",
    "has_second_vehicle",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

READABLE_NAMES = {
    "age": "Age of main driver",
    "daily_commute_km": "Daily commute (km)",
    "current_vehicle_age_years": "Age of current vehicle (years)",
    "monthly_fuel_cost_nu": "Monthly fuel cost (Nu)",
    "electricity_hours_per_day": "Reliable electricity (hours/day)",
    "dzongkhag": "Dzongkhag",
    "area_type": "Area type",
    "income_band": "Household income band",
    "charging_access": "Charging access",
    "terrain": "Terrain around home",
    "knows_about_subsidy": "Knows about EV subsidy",
    "has_second_vehicle": "Owns a second vehicle",
}


# ----------------------------------------------------------------------
# Step 1: build the synthetic dataset
# ----------------------------------------------------------------------


def _standardise(values):
    """Turn a numeric array into z-scores so weights are comparable."""
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


@st.cache_data
def generate_dataset(n_records=N_RECORDS, seed=RANDOM_SEED):
    """Create a fictional dataset of households and their EV intention."""
    rng = np.random.default_rng(seed)

    # --- Features -----------------------------------------------------
    age = rng.integers(22, 66, n_records)
    daily_commute_km = np.round(np.clip(rng.gamma(2.2, 7.0, n_records), 1, 90), 1)
    current_vehicle_age_years = rng.integers(0, 21, n_records)
    electricity_hours_per_day = np.round(np.clip(rng.normal(22.5, 1.6, n_records), 16, 24), 1)

    dzongkhag = rng.choice(
        DZONGKHAGS,
        n_records,
        p=[0.30, 0.13, 0.12, 0.10, 0.09, 0.07, 0.10, 0.09],
    )
    area_type = rng.choice(AREA_TYPES, n_records, p=[0.45, 0.30, 0.25])
    income_band = rng.choice(INCOME_BANDS, n_records, p=[0.28, 0.34, 0.24, 0.14])
    charging_access = rng.choice(CHARGING_OPTIONS, n_records, p=[0.33, 0.30, 0.37])
    terrain = rng.choice(TERRAIN_TYPES, n_records, p=[0.38, 0.40, 0.22])
    knows_about_subsidy = rng.choice(YES_NO, n_records, p=[0.55, 0.45])
    has_second_vehicle = rng.choice(YES_NO, n_records, p=[0.30, 0.70])

    # Fuel cost grows with commute distance, plus some household variation
    monthly_fuel_cost_nu = np.round(
        np.clip(daily_commute_km * 95 + rng.normal(1800, 900, n_records), 600, 18000),
        0,
    )

    data = pd.DataFrame(
        {
            "age": age,
            "daily_commute_km": daily_commute_km,
            "current_vehicle_age_years": current_vehicle_age_years,
            "monthly_fuel_cost_nu": monthly_fuel_cost_nu,
            "electricity_hours_per_day": electricity_hours_per_day,
            "dzongkhag": dzongkhag,
            "area_type": area_type,
            "income_band": income_band,
            "charging_access": charging_access,
            "terrain": terrain,
            "knows_about_subsidy": knows_about_subsidy,
            "has_second_vehicle": has_second_vehicle,
        }
    )

    # --- Fictional scoring rule --------------------------------------
    # Each line is an assumption I chose, not a measured fact.
    income_rank = data["income_band"].map({band: i for i, band in enumerate(INCOME_BANDS)})

    score = (
        1.00 * (data["charging_access"] == "Home charging possible")
        + 0.45 * (data["charging_access"] == "Public station nearby")
        + 0.42 * income_rank
        + 0.55 * (data["area_type"] == "Urban")
        + 0.25 * (data["area_type"] == "Semi-urban")
        + 0.45 * (data["dzongkhag"].isin(["Thimphu", "Paro"]))
        + 0.50 * (data["knows_about_subsidy"] == "Yes")
        + 0.35 * (data["has_second_vehicle"] == "Yes")
        + 0.30 * (data["terrain"] == "Flat valley")
        - 0.30 * (data["terrain"] == "Steep mountain")
        + 0.55 * _standardise(data["monthly_fuel_cost_nu"])
        + 0.40 * _standardise(data["current_vehicle_age_years"])
        - 0.60 * _standardise(data["daily_commute_km"])
        + 0.25 * _standardise(data["electricity_hours_per_day"])
        - 0.20 * _standardise(data["age"])
        + rng.normal(0, 0.85, n_records)  # noise, so the task is not trivial
    )

    # The most positive 40% of scores are labelled "Likely to adopt"
    cutoff = np.quantile(score, 0.60)
    data[TARGET_NAME] = np.where(score >= cutoff, POSITIVE_LABEL, NEGATIVE_LABEL)

    # --- Add a few missing values so the pipeline has to handle them ---
    for column in [
        "daily_commute_km",
        "monthly_fuel_cost_nu",
        "charging_access",
        "electricity_hours_per_day",
    ]:
        missing_rows = rng.choice(n_records, size=int(0.02 * n_records), replace=False)
        data.loc[missing_rows, column] = np.nan

    return data


# ----------------------------------------------------------------------
# Step 2: build and train the model
# ----------------------------------------------------------------------


@st.cache_resource
def train_model(data):
    """Train logistic regression and return the model plus its test results."""
    X = data[ALL_FEATURES]
    y = (data[TARGET_NAME] == POSITIVE_LABEL).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Numeric columns: fill gaps with the median, then put on the same scale
    numeric_steps = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # Categorical columns: fill gaps with the most common value, then one-hot encode
    categorical_steps = Pipeline(
        [
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_steps, NUMERIC_FEATURES),
            ("categorical", categorical_steps, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        [
            ("prepare", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, target_names=[NEGATIVE_LABEL, POSITIVE_LABEL]
        ),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, results


def coefficient_table(model):
    """Pull the logistic regression weights out of the fitted pipeline."""
    preprocessor = model.named_steps["prepare"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encode"]

    names = list(NUMERIC_FEATURES) + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    weights = model.named_steps["classifier"].coef_[0]

    table = pd.DataFrame({"feature": names, "weight": weights})
    table["effect"] = np.where(table["weight"] >= 0, "Pushes towards adopting", "Pushes against")
    return table.sort_values("weight", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Load everything once
#
# st.spinner shows a rotating indicator while this runs. Because both
# functions are cached, it only appears the first time the app opens.
# ----------------------------------------------------------------------

with st.spinner("Building the fictional dataset and training the model..."):
    df = generate_dataset()
    pipeline, results = train_model(df)

# ----------------------------------------------------------------------
# Hero banner
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">BHUTAN &nbsp;·&nbsp; MACHINE LEARNING PROJECT</div>
        <h1 class="hero-title">GOING<br>ELECTRIC</h1>
        <div class="hero-rule"></div>
        <p class="hero-sub">
            Bhutan runs on hydropower but imports every litre of fuel it burns.
            This project asks which households are most likely to make the switch
            to an electric vehicle &mdash; and builds a small machine learning
            model to find out. All data shown here is fictional.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_explore, tab_predict, tab_results = st.tabs(
    ["Overview", "Explore data", "Prediction demo", "Model results"]
)

# ----------------------------------------------------------------------
# Tab 1: Overview
# ----------------------------------------------------------------------

with tab_overview:
    st.header("About this project")

    st.markdown(
        """
Bhutan generates most of its electricity from hydropower but imports every litre
of fuel it burns. That makes electric vehicles an unusually interesting question
here: the "fuel" is already domestic, while the hills, the long inter-dzongkhag
drives and the cost of a new car all push the other way.

This app asks a simple version of that question:

> **Given what we know about a household, is it likely to switch to an electric
> vehicle in the next two years?**
        """
    )

    st.subheader("⚠️ Please read: the data is invented")
    st.warning(
        "Every row in this app was generated by a formula I wrote myself. "
        "No survey was carried out and no real household, person or vehicle is "
        "represented. The numbers show how the method works, not what Bhutanese "
        "households actually intend to do. Nothing here should be used for policy, "
        "business or purchasing decisions."
    )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("What the model uses")
            st.markdown(
                """
**Numeric inputs**
- Age of the main driver
- Daily commute distance (km)
- Age of the current vehicle (years)
- Monthly fuel spending (Nu)
- Reliable electricity per day (hours)

**Category inputs**
- Dzongkhag
- Area type (urban, semi-urban, rural)
- Household income band
- Charging access
- Terrain around the home
- Awareness of the EV subsidy
- Whether a second vehicle is owned
                """
            )

    with right:
        with st.container(border=True):
            st.subheader("How it was built")
            st.markdown(
                f"""
1. Generate {N_RECORDS} fictional households with a fixed seed ({RANDOM_SEED}),
   so the results are the same every time the app runs.
2. Score each household with a weighted rule, add random noise, and label the
   top 40% as **{POSITIVE_LABEL}**.
3. Remove about 2% of values in four columns, so the pipeline has to deal with
   missing data the way a real dataset would.
4. Split 80% for training and 20% for testing.
5. Fill gaps, scale numbers, one-hot encode categories, then fit a logistic
   regression.
6. Score the model on the held-out test set.
                """
            )

    st.subheader("An honest limitation")
    st.info(
        "The model can only rediscover the rule I used to build the data. A good "
        "score here shows the pipeline works end to end; it says nothing about "
        "whether the rule itself matches reality. Testing that would need a real survey."
    )

# ----------------------------------------------------------------------
# Tab 2: Explore data
# ----------------------------------------------------------------------

with tab_explore:
    st.header("Explore the fictional dataset")

    adoption_rate = (df[TARGET_NAME] == POSITIVE_LABEL).mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Households", f"{len(df):,}")
    col2.metric("Features", len(ALL_FEATURES))
    col3.metric(POSITIVE_LABEL, f"{adoption_rate:.0%}")
    col4.metric("Missing values", f"{int(df.isna().sum().sum())}")

    st.divider()

    st.subheader("Distribution of one feature")
    chosen = st.selectbox(
        "Pick a feature to look at",
        ALL_FEATURES,
        format_func=lambda name: READABLE_NAMES[name],
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        fig, ax = plt.subplots(figsize=(6, 4))
        if chosen in NUMERIC_FEATURES:
            sns.histplot(df[chosen].dropna(), bins=25, kde=True, color=CYAN, ax=ax)
        else:
            order = df[chosen].value_counts().index
            sns.countplot(y=df[chosen], order=order, color=CYAN, ax=ax)
        ax.set_title(f"Distribution of {READABLE_NAMES[chosen]}")
        ax.set_xlabel(READABLE_NAMES[chosen] if chosen in NUMERIC_FEATURES else "Households")
        ax.set_ylabel("Households" if chosen in NUMERIC_FEATURES else "")
        st.pyplot(fig, transparent=True)
        plt.close(fig)

    with chart_right:
        fig, ax = plt.subplots(figsize=(6, 4))
        if chosen in NUMERIC_FEATURES:
            sns.boxplot(
                data=df,
                x=TARGET_NAME,
                y=chosen,
                hue=TARGET_NAME,
                order=[NEGATIVE_LABEL, POSITIVE_LABEL],
                palette=[TEAL, CYAN],
                legend=False,
                ax=ax,
            )
            ax.set_title(f"{READABLE_NAMES[chosen]} by group")
            ax.set_xlabel("")
            ax.set_ylabel(READABLE_NAMES[chosen])
        else:
            rate = (
                df.groupby(chosen, observed=True)[TARGET_NAME]
                .apply(lambda s: (s == POSITIVE_LABEL).mean())
                .sort_values()
            )
            sns.barplot(x=rate.to_numpy(), y=rate.index.astype(str), color=CYAN, ax=ax)
            ax.set_title(f"Share likely to adopt, by {READABLE_NAMES[chosen].lower()}")
            ax.set_xlabel("Share likely to adopt")
            ax.set_ylabel("")
        st.pyplot(fig, transparent=True)
        plt.close(fig)

    st.divider()

    st.subheader("How the two groups compare")
    numeric_summary = (
        df.groupby(TARGET_NAME, observed=True)[NUMERIC_FEATURES].mean().round(1).T
    )
    numeric_summary.index = [READABLE_NAMES[name] for name in numeric_summary.index]
    st.dataframe(numeric_summary)

    st.caption(
        "Households labelled likely to adopt tend to drive shorter distances but "
        "spend more on fuel relative to that distance, and they own older vehicles."
    )

    st.divider()

    st.subheader("The fictional records")
    show_only = st.radio(
        "Show",
        ["All households", POSITIVE_LABEL, NEGATIVE_LABEL],
        horizontal=True,
    )
    table = df if show_only == "All households" else df[df[TARGET_NAME] == show_only]
    st.dataframe(table.head(100))
    st.caption(f"Showing the first {min(100, len(table))} of {len(table)} fictional rows.")

# ----------------------------------------------------------------------
# Tab 3: Prediction demo
# ----------------------------------------------------------------------

with tab_predict:
    st.header("Try a fictional household")
    st.write(
        "Set the values below and the trained model will estimate how likely that "
        "household is to switch to an electric vehicle. This is a demonstration "
        "of the model, not advice."
    )

    input_left, input_middle, input_right = st.columns(3)

    with input_left:
        age = st.slider("Age of main driver", 22, 65, 35)
        daily_commute_km = st.slider("Daily commute (km)", 1.0, 90.0, 18.0, step=0.5)
        current_vehicle_age_years = st.slider("Age of current vehicle (years)", 0, 20, 7)
        monthly_fuel_cost_nu = st.slider("Monthly fuel cost (Nu)", 600, 18000, 4500, step=100)

    with input_middle:
        electricity_hours_per_day = st.slider(
            "Reliable electricity (hours/day)", 16.0, 24.0, 22.5, step=0.5
        )
        # The default choices below describe a fairly typical household,
        # so the demo does not open on an extreme prediction.
        dzongkhag = st.selectbox("Dzongkhag", DZONGKHAGS, index=0)
        area_type = st.selectbox("Area type", AREA_TYPES, index=1)
        income_band = st.selectbox("Household income band", INCOME_BANDS, index=1)

    with input_right:
        charging_access = st.selectbox("Charging access", CHARGING_OPTIONS, index=1)
        terrain = st.selectbox("Terrain around home", TERRAIN_TYPES, index=1)
        knows_about_subsidy = st.selectbox("Knows about the EV subsidy", YES_NO, index=1)
        has_second_vehicle = st.selectbox("Owns a second vehicle", YES_NO, index=1)

    one_household = pd.DataFrame(
        [
            {
                "age": age,
                "daily_commute_km": daily_commute_km,
                "current_vehicle_age_years": current_vehicle_age_years,
                "monthly_fuel_cost_nu": monthly_fuel_cost_nu,
                "electricity_hours_per_day": electricity_hours_per_day,
                "dzongkhag": dzongkhag,
                "area_type": area_type,
                "income_band": income_band,
                "charging_access": charging_access,
                "terrain": terrain,
                "knows_about_subsidy": knows_about_subsidy,
                "has_second_vehicle": has_second_vehicle,
            }
        ]
    )

    probability = float(pipeline.predict_proba(one_household)[0, 1])
    predicted_label = POSITIVE_LABEL if probability >= 0.5 else NEGATIVE_LABEL

    st.divider()

    result_left, result_right = st.columns([1, 2])

    with result_left:
        if predicted_label == POSITIVE_LABEL:
            st.success(f"Prediction: **{POSITIVE_LABEL}**")
        else:
            st.info(f"Prediction: **{NEGATIVE_LABEL}**")
        st.metric("Estimated probability of adopting", f"{probability:.1%}")

    with result_right:
        # Fill the bar in small steps so the result animates into place.
        # The whole loop takes well under half a second.
        target_percent = int(round(probability * 100))
        progress_bar = st.progress(0)
        for filled in range(0, target_percent + 1, 4):
            progress_bar.progress(filled / 100)
            time.sleep(0.008)
        progress_bar.progress(target_percent / 100)

        st.caption(
            "The bar shows the model's estimated probability. Anything at or above "
            "50% is reported as likely to adopt."
        )
        st.caption(
            "Remember that the model learned from invented data, so this is a "
            "demonstration of the method and not a real forecast."
        )

# ----------------------------------------------------------------------
# Tab 4: Model results
# ----------------------------------------------------------------------

with tab_results:
    st.header("How well did the model do?")
    st.caption(
        f"Trained on {results['n_train']} households and tested on "
        f"{results['n_test']} the model never saw during training."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{results['accuracy']:.3f}")
    m2.metric("ROC-AUC", f"{results['roc_auc']:.3f}")
    m3.metric("Precision", f"{results['precision']:.3f}")
    m4.metric("Recall", f"{results['recall']:.3f}")
    m5.metric("F1-score", f"{results['f1']:.3f}")

    st.divider()

    matrix_col, report_col = st.columns(2)

    with matrix_col:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            results["confusion"],
            annot=True,
            fmt="d",
            cmap="mako",
            cbar=False,
            linewidths=1,
            linecolor=INK,
            xticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
            yticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
            ax=ax,
        )
        ax.set_xlabel("Model predicted")
        ax.set_ylabel("Actual label")
        ax.grid(False)
        st.pyplot(fig, transparent=True)
        plt.close(fig)
        st.caption(
            "Numbers on the diagonal are correct predictions. The top-right cell is "
            "households the model wrongly flagged as adopters."
        )

    with report_col:
        st.subheader("Classification report")
        st.code(results["report"])
        st.caption(
            "Precision asks: of the households flagged as adopters, how many really "
            "were? Recall asks: of the real adopters, how many did we catch?"
        )

    st.divider()

    st.subheader("Which inputs mattered most?")
    weights = coefficient_table(pipeline)
    top_weights = pd.concat([weights.head(8), weights.tail(8)])

    fig, ax = plt.subplots(figsize=(9, 6))
    colours = [CYAN if w >= 0 else CORAL for w in top_weights["weight"]]
    ax.barh(top_weights["feature"], top_weights["weight"], color=colours)
    ax.axvline(0, color=LIGHT, linewidth=0.8)
    ax.set_xlabel("Logistic regression weight")
    ax.set_title("Strongest pushes towards (cyan) and against (red) adoption")
    ax.invert_yaxis()
    st.pyplot(fig, transparent=True)
    plt.close(fig)

    st.caption(
        "These weights come from the model, so they reflect the rule used to build "
        "the fictional data. On real data they would need to be interpreted far more carefully."
    )

    with st.expander("See every weight"):
        st.dataframe(weights)
