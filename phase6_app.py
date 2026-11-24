# ============================================================
# ShopSurvival — Phase 6: Streamlit Web App
# Run with: streamlit run phase6_app.py
# ============================================================

from collections import Counter
from pathlib import Path
import re

import joblib  # Kept available for compatibility with the project environment.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="ShopSurvival — Business Survival Predictor",
    page_icon="🏪",
    layout="wide",
)

# ------------------------- THEME -------------------------------
BG = "#171614"
AXES_BG = "#242321"
TEXT = "#FFFFFF"
MUTED = "#D0CEC9"
GRID = "#77736D"
GREEN = "#5DCAA5"
YELLOW = "#FAC775"
RED = "#F0997B"
PURPLE = "#AFA9EC"

plt.rcParams.update({
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "axes.edgecolor": MUTED,
    "figure.facecolor": BG,
    "axes.facecolor": AXES_BG,
    "savefig.facecolor": BG,
})


def style_axis(ax, *, grid=False):
    """Apply readable white labels/ticks to charts on Streamlit's dark theme."""
    ax.set_facecolor(AXES_BG)
    ax.tick_params(axis="both", colors=TEXT, labelcolor=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    if grid:
        ax.grid(True, color=GRID, alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)


def finish_chart(fig):
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ------------------------- DATA --------------------------------
@st.cache_data

def load_data():
    data_path = Path("data/clustered_businesses.csv")
    if data_path.exists():
        data = pd.read_csv(data_path)
    else:
        # Demonstration data only; replace with the real prepared dataset for real analysis.
        rng = np.random.default_rng(42)
        n = 2000
        categories = [
            "Restaurants", "Shopping", "Beauty & Spas", "Health",
            "Coffee & Tea", "Bars", "Retail", "Automotive",
        ]
        cities = ["Phoenix", "Las Vegas", "Charlotte", "Pittsburgh", "Cleveland"]
        data = pd.DataFrame({
            "name": [f"Business {i}" for i in range(n)],
            "city": rng.choice(cities, n),
            "primary_category": rng.choice(categories, n),
            "stars": np.round(rng.uniform(1, 5, n) * 2) / 2,
            "review_count": rng.integers(1, 2000, n),
            "log_reviews": np.log1p(rng.integers(1, 2000, n)),
            "competition_density": rng.integers(5, 200, n),
            "stars_zscore": rng.normal(0, 1, n).round(2),
            "rating_bucket": rng.integers(1, 5, n),
            "survived": rng.choice([0, 1], n, p=[0.2, 0.8]),
            "cluster": rng.integers(0, 4, n),
        })

    required = ["city", "primary_category", "stars", "survived"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )

    # Normalize numeric columns and create derived fields when absent.
    for col in ["stars", "survived", "review_count", "log_reviews",
                "competition_density", "stars_zscore", "rating_bucket", "cluster"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "review_count" not in data.columns:
        data["review_count"] = 0
    if "log_reviews" not in data.columns:
        data["log_reviews"] = np.log1p(data["review_count"].clip(lower=0))
    if "competition_density" not in data.columns:
        data["competition_density"] = 0
    if "stars_zscore" not in data.columns:
        grouped = data.groupby("primary_category")["stars"]
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0, np.nan)
        data["stars_zscore"] = ((data["stars"] - means) / stds).fillna(0)
    if "rating_bucket" not in data.columns:
        data["rating_bucket"] = np.digitize(data["stars"].fillna(0), [2.5, 3.5, 4.5])
    if "cluster" not in data.columns:
        data["cluster"] = 0

    data["city"] = data["city"].fillna("Unknown").astype(str)
    data["primary_category"] = data["primary_category"].fillna("Unknown").astype(str)
    for col in ["stars", "log_reviews", "competition_density", "stars_zscore", "rating_bucket"]:
        data[col] = data[col].fillna(data[col].median() if data[col].notna().any() else 0)
    data["survived"] = data["survived"].fillna(0).astype(int).clip(0, 1)
    data["cluster"] = data["cluster"].fillna(0).astype(int)
    return data


@st.cache_resource

def load_models(data):
    model_data = data.copy()
    category_encoder = LabelEncoder()
    city_encoder = LabelEncoder()
    model_data["category_enc"] = category_encoder.fit_transform(
        model_data["primary_category"].astype(str)
    )
    model_data["city_enc"] = city_encoder.fit_transform(model_data["city"].astype(str))

    feature_cols = [
        "stars", "log_reviews", "competition_density", "stars_zscore",
        "rating_bucket", "category_enc", "city_enc",
    ]
    X = model_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = model_data["survived"]
    if y.nunique() < 2:
        raise ValueError("The 'survived' column must contain both 0 and 1 classes to train the prediction model.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_scaled, y)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)
    return model, scaler, pca, category_encoder, city_encoder, feature_cols


try:
    df = load_data()
    rf, scaler, pca, category_encoder, city_encoder, feature_cols = load_models(df)
except Exception as exc:
    st.error(f"Could not load the project data or train the model: {exc}")
    st.stop()

CLUSTER_NAMES = {
    0: ("Struggling Shops", "⚠️", RED),
    1: ("Steady Locals", "🟡", YELLOW),
    2: ("Popular Hubs", "✅", GREEN),
    3: ("Competitive Hotspots", "🔵", PURPLE),
}

# ------------------------- SIDEBAR -----------------------------
st.sidebar.title("🏪 ShopSurvival")
st.sidebar.caption("AI Business Survival Predictor")
tab_choice = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Explore", "🔮 Predict", "🔵 Clusters", "💬 Review Brain"],
)
categories = sorted(df["primary_category"].unique().tolist())
cities = sorted(df["city"].unique().tolist())

# ------------------------- OVERVIEW ----------------------------
if tab_choice == "🏠 Overview":
    st.title("🏪 ShopSurvival")
    st.subheader("Will your business survive? Let data decide.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Businesses", f"{len(df):,}")
    col2.metric("Survival Rate", f"{df['survived'].mean() * 100:.1f}%")
    col3.metric("Avg Rating", f"{df['stars'].mean():.2f} ⭐")
    col4.metric("Categories", int(df["primary_category"].nunique()))

    st.markdown("---")
    st.markdown("""
    **What this app does:**
    - 📊 **Explore** — visualise survival patterns across cities and categories
    - 🔮 **Predict** — enter your business details and get a model-estimated survival probability
    - 🔵 **Clusters** — inspect business groups using PCA and the available cluster labels
    - 💬 **Review Brain** — paste reviews for a simple keyword-based sentiment check
    """)

    st.markdown("#### Survival rate by category")
    surv = (
        df.groupby("primary_category")["survived"]
        .agg(["mean", "count"])
        .query("count >= 1")
        .sort_values("mean", ascending=False)
        .head(10)
    )
    if surv.empty:
        st.info("There is not enough data to display this chart.")
    else:
        surv["mean"] *= 100
        fig, ax = plt.subplots(figsize=(9, 4))
        bar_colors = [GREEN if v > 80 else YELLOW if v > 65 else RED for v in surv["mean"]]
        ax.barh(surv.index, surv["mean"], color=bar_colors)
        ax.invert_yaxis()
        ax.set_xlabel("Survival Rate (%)")
        ax.set_ylabel("Business Category")
        ax.axvline(surv["mean"].mean(), color=TEXT, linestyle="--", alpha=0.75,
                   label=f"Average: {surv['mean'].mean():.1f}%")
        ax.legend(facecolor=AXES_BG, edgecolor=MUTED, labelcolor=TEXT)
        style_axis(ax, grid=True)
        finish_chart(fig)

# ------------------------- EXPLORE -----------------------------
elif tab_choice == "📊 Explore":
    st.title("📊 Explore Survival Patterns")
    col1, col2 = st.columns(2)
    selected_city = col1.selectbox("Filter by city", ["All"] + cities)
    selected_cat = col2.selectbox("Filter by category", ["All"] + categories)

    filtered = df.copy()
    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]
    if selected_cat != "All":
        filtered = filtered[filtered["primary_category"] == selected_cat]
    st.metric("Businesses shown", len(filtered))

    if filtered.empty:
        st.warning("No businesses match these filters. Choose another city or category.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Rating distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for label, color, name in [(1, GREEN, "Survived"), (0, RED, "Closed")]:
                values = filtered.loc[filtered["survived"] == label, "stars"]
                if not values.empty:
                    ax.hist(values, bins=15, alpha=0.65, color=color,
                            label=name, density=True)
            ax.set_xlabel("Stars")
            ax.set_ylabel("Density")
            ax.legend(facecolor=AXES_BG, edgecolor=MUTED, labelcolor=TEXT)
            style_axis(ax, grid=True)
            finish_chart(fig)

        with col2:
            st.markdown("#### Correlation heatmap")
            num_cols = ["stars", "log_reviews", "competition_density", "survived"]
            corr = filtered[num_cols].corr()
            fig, ax = plt.subplots(figsize=(5.5, 4))
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, linewidths=0.5, annot_kws={"color": "black"},
                cbar_kws={"label": "Correlation"},
            )
            ax.tick_params(axis="both", colors=TEXT, labelcolor=TEXT)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color=TEXT)
            plt.setp(cbar.ax.get_yticklabels(), color=TEXT)
            cbar.set_label("Correlation", color=TEXT)
            style_axis(ax)
            finish_chart(fig)

        st.markdown("#### Competition density: survived vs closed")
        box_data = filtered.copy()
        box_data["Status"] = box_data["survived"].map({1: "Survived", 0: "Closed"})
        fig, ax = plt.subplots(figsize=(9, 3.5))
        sns.boxplot(
            data=box_data, x="Status", y="competition_density",
            hue="Status", palette={"Survived": GREEN, "Closed": RED},
            legend=False, ax=ax,
        )
        ax.set_xlabel("Business status")
        ax.set_ylabel("Nearby competition density")
        style_axis(ax, grid=True)
        finish_chart(fig)

# ------------------------- PREDICT -----------------------------
elif tab_choice == "🔮 Predict":
    st.title("🔮 Predict Business Survival")
    st.caption("Enter business details to get a model-estimated survival probability.")

    col1, col2 = st.columns(2)
    with col1:
        b_name = st.text_input("Business name", "My New Cafe")
        b_category = st.selectbox("Category", categories)
        b_city = st.selectbox("City", cities)
    with col2:
        b_stars = st.slider("Expected star rating", 1.0, 5.0, 4.0, 0.5)
        b_reviews = st.slider("Estimated monthly reviews", 1, 500, 50)
        b_comp = st.slider("Nearby competitors (same category in city)", 5, 300, 60)

    if st.button("🔮 Predict survival probability", use_container_width=True):
        log_rev = float(np.log1p(b_reviews))
        cat_rows = df[df["primary_category"] == b_category]["stars"]
        avg_stars = float(cat_rows.mean()) if not cat_rows.empty else float(df["stars"].mean())
        std_stars = float(cat_rows.std()) if cat_rows.std() and cat_rows.std() > 0 else 1.0
        zscore = (b_stars - avg_stars) / std_stars
        rating_bucket = int(np.digitize(b_stars, [2.5, 3.5, 4.5]))

        cat_enc = int(category_encoder.transform([b_category])[0])
        city_enc = int(city_encoder.transform([b_city])[0])
        features = pd.DataFrame([{
            "stars": b_stars,
            "log_reviews": log_rev,
            "competition_density": b_comp,
            "stars_zscore": round(zscore, 2),
            "rating_bucket": rating_bucket,
            "category_enc": cat_enc,
            "city_enc": city_enc,
        }], columns=feature_cols)
        probability = float(rf.predict_proba(scaler.transform(features))[0, 1])

        st.markdown("---")
        if probability >= 0.75:
            verdict = "✅ Strong survival outlook"
        elif probability >= 0.50:
            verdict = "🟡 Moderate outlook — some risk factors are present"
        else:
            verdict = "🔴 Higher risk — consider adjusting your strategy"
        st.markdown(f"### {verdict}")
        st.progress(probability)
        st.markdown(f"**{probability * 100:.1f}% model-estimated survival probability** for **{b_name}**")
        st.caption("This is an estimate from the available dataset, not a guarantee of future business performance.")

        st.markdown("#### Most influential model features")
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=True).tail(5)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color=PURPLE)
        ax.set_xlabel("Random Forest feature importance")
        style_axis(ax, grid=True)
        finish_chart(fig)
        st.info(f"Most influential feature in this model: **{importance_df.iloc[-1]['Feature']}**.")

# ------------------------- CLUSTERS ----------------------------
elif tab_choice == "🔵 Clusters":
    st.title("🔵 Business Personality Clusters")
    st.caption("PCA view of the available business cluster labels")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Business groups")
        for cluster_id, (name, icon, color) in CLUSTER_NAMES.items():
            cluster_data = df[df["cluster"] == cluster_id]
            if cluster_data.empty:
                st.markdown(f"**{icon} {name}**\n\nNo records in this group.")
            else:
                survival = cluster_data["survived"].mean() * 100
                avg_rating = cluster_data["stars"].mean()
                st.markdown(
                    f"**{icon} {name}**  \n"
                    f"Businesses: {len(cluster_data):,}  \n"
                    f"Survival: {survival:.1f}% | Avg rating: {avg_rating:.1f} ⭐"
                )
            st.markdown("---")

    with col2:
        st.markdown("#### PCA scatter — 2D view of businesses")
        model_data = df.copy()
        model_data["category_enc"] = category_encoder.transform(model_data["primary_category"].astype(str))
        model_data["city_enc"] = city_encoder.transform(model_data["city"].astype(str))
        X = model_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        coords = pca.transform(scaler.transform(X))
        fig, ax = plt.subplots(figsize=(7, 5))
        cluster_colors = [RED, YELLOW, GREEN, PURPLE]
        for cluster_id in sorted(model_data["cluster"].unique()):
            mask = model_data["cluster"].to_numpy() == cluster_id
            color = cluster_colors[cluster_id % len(cluster_colors)]
            label = CLUSTER_NAMES.get(cluster_id, (f"Cluster {cluster_id}", "", color))[0]
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                       label=label, alpha=0.55, s=14)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(fontsize=8, facecolor=AXES_BG, edgecolor=MUTED, labelcolor=TEXT)
        style_axis(ax, grid=True)
        finish_chart(fig)

# ------------------------- REVIEW BRAIN ------------------------
elif tab_choice == "💬 Review Brain":
    st.title("💬 Review Brain — Sentiment Analyser")
    st.caption("Paste customer reviews to run a simple keyword-based sentiment check.")
    reviews_text = st.text_area(
        "Paste reviews here (one per line)",
        placeholder=("Great food, came back 3 times already!\n"
                     "Terrible service, waited 45 minutes.\n"
                     "Decent place, nothing special."),
        height=180,
    )

    if st.button("🔍 Analyse reviews", use_container_width=True):
        lines = [line.strip() for line in reviews_text.splitlines() if line.strip()]
        if not lines:
            st.warning("Please enter at least one review.")
        else:
            positive_words = {
                "great", "amazing", "best", "love", "excellent", "fantastic",
                "good", "wonderful", "recommend", "friendly", "clean", "fast",
                "fresh", "tasty", "delicious",
            }
            negative_words = {
                "terrible", "worst", "awful", "bad", "slow", "rude", "dirty",
                "cold", "overpriced", "never", "disappointing", "horrible",
                "disgusting", "closed", "avoid",
            }
            results = []
            for review in lines:
                words = set(re.findall(r"[a-z']+", review.lower()))
                positive = len(words & positive_words)
                negative = len(words & negative_words)
                if positive > negative:
                    label, color = "Positive ✅", GREEN
                elif negative > positive:
                    label, color = "Negative ❌", RED
                else:
                    label, color = "Neutral 😐", YELLOW
                results.append((review, label, color))

            positive_count = sum(label.startswith("Positive") for _, label, _ in results)
            negative_count = sum(label.startswith("Negative") for _, label, _ in results)
            total = len(results)
            positive_percent = positive_count / total * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive reviews", f"{positive_count}/{total}")
            col2.metric("Negative reviews", f"{negative_count}/{total}")
            col3.metric("Positive review share", f"{positive_percent:.0f}%")

            if positive_percent >= 70:
                st.success("🎉 Mostly positive sentiment in the submitted reviews.")
            elif positive_percent >= 40:
                st.warning("⚠️ Mixed sentiment — review the negative feedback.")
            else:
                st.error("🚨 Mostly neutral or negative sentiment in the submitted reviews.")

            st.markdown("#### Individual review analysis")
            for review, label, color in results:
                st.markdown(f"- :{ 'green' if color == GREEN else 'red' if color == RED else 'orange'}[**{label}**] — _{review}_")

            all_words = re.findall(r"[a-z']+", " ".join(lines).lower())
            stopwords = {"the", "a", "is", "was", "and", "to", "in", "it", "for", "but", "i"}
            top_words = [word for word, _ in Counter(all_words).most_common(30)
                         if word not in stopwords][:8]
            st.markdown("**Most mentioned:** " + (", ".join(top_words) if top_words else "No common words found."))

# ------------------------- FOOTER ------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · Scikit-learn · Streamlit")
st.sidebar.caption("Data: project business dataset")
