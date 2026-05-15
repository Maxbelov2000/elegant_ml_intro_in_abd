import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import urllib3
import re
import random
from itertools import combinations

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error as MSE

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#  Воспроизводимость 
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

CARS_TRAIN = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_train.csv'
CARS_TEST  = 'https://github.com/evgpat/datasets/raw/refs/heads/main/cars_test.csv'


# Конфиг страницы


st.set_page_config(
    page_title="Car Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #2563eb;
        padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin-bottom: 0.5rem;
    }
    .metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #2563eb; }
    .metric-label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    .prediction-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        color: white; padding: 2rem; border-radius: 12px; text-align: center; margin: 1rem 0;
    }
    .prediction-price { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


#  Загрузка данных


def read_csv_secure(url):
    """Безопасное чтение CSV по URL"""
    response = requests.get(url, verify=False)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


@st.cache_data
def load_data():
    df_train = read_csv_secure(CARS_TRAIN)
    df_test  = read_csv_secure(CARS_TEST)
    return df_train, df_test


# Предобработка числовых признаков

def parse_torque(val):
    """Парсинг признака torque """
    if pd.isna(val):
        return np.nan, np.nan
    val = val.strip().lower()
    is_kgm = "kgm" in val
    torque_match = re.search(r"([\d.]+)", val)
    torque_val = float(torque_match.group(1)) if torque_match else np.nan
    if is_kgm and not np.isnan(torque_val):
        torque_val *= 9.80665
    rpm_match = re.search(r"[@+/]\s*([\d,]+)", val)
    if not rpm_match:
        rpm_match = re.search(r"([\d,]+)\s*rpm", val)
    rpm_val = float(rpm_match.group(1).replace(",", "")) if rpm_match else np.nan
    return torque_val, rpm_val


def preprocess_features(df):
    """Парсинг строковых признаков в числа """
    df = df.copy()
    df["mileage"]   = df["mileage"].str.extract(r"([\d.]+)").astype(float)
    df["engine"]    = df["engine"].str.extract(r"([\d.]+)").astype(float)
    df["max_power"] = df["max_power"].str.extract(r"([\d.]+)").astype(float)
    torque_parsed = df["torque"].apply(parse_torque)
    df["torque"]         = torque_parsed.apply(lambda x: x[0])
    df["max_torque_rpm"] = torque_parsed.apply(lambda x: x[1])
    return df


# Предобработка признака name


def add_brand(df):
    """Извлекаем марку из названия"""
    df = df.copy()
    df["brand"] = df["name"].str.split().str[0]
    return df


# R² вручную

def r2_manual(y_true, y_pred):
    """R² реализованный вручную"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


# Бизнес-метрика

def business_metric(y_true, y_pred):
    """Доля предсказаний с ошибкой не более 10% """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    relative_error = np.abs(y_pred - y_true) / y_true
    return (relative_error <= 0.10).mean()


# Асимметричная метрика

def asymmetric_metric(y_true, y_pred, penalty_under=2.0):
    """
    Асимметричная метрика: недопрогноз штрафуется вдвое.
    Чем меньше — тем лучше.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    relative_error = (y_pred - y_true) / y_true
    loss = np.where(
        relative_error < 0,
        penalty_under * np.abs(relative_error),
        np.abs(relative_error)
    )
    return loss.mean()



# Сбор метрик 

def get_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test):
    return {
        "R² Train":         round(r2_score(y_true_train, y_pred_train), 4),
        "R² Test":          round(r2_score(y_true_test,  y_pred_test),  4),
        "RMSE Train":       int(np.sqrt(MSE(y_true_train, y_pred_train))),
        "RMSE Test":        int(np.sqrt(MSE(y_true_test,  y_pred_test))),
        "Business Train":   round(business_metric(y_true_train, y_pred_train), 4),
        "Business Test":    round(business_metric(y_true_test,  y_pred_test),  4),
        "Asymmetric Train": round(asymmetric_metric(y_true_train, y_pred_train), 4),
        "Asymmetric Test":  round(asymmetric_metric(y_true_test,  y_pred_test),  4),
    }


# Полный пайплайн предобработки

@st.cache_data
def full_pipeline(df_train_raw, df_test_raw):
    # Задание 4: парсинг числовых признаков
    df_train = preprocess_features(df_train_raw)
    df_test  = preprocess_features(df_test_raw)

    # Задание 19: марка авто
    df_train = add_brand(df_train)
    df_test  = add_brand(df_test)

    # Задание 3: удаление дубликатов по признакам (без таргета)
    target    = "selling_price"
    feat_cols = [c for c in df_train.columns if c != target]
    df_train  = df_train.drop_duplicates(subset=feat_cols, keep="first")
    df_train  = df_train.reset_index(drop=True)

    # Задание 5: заполнение пропусков медианами из train
    num_cols = df_train.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != target]
    train_medians = df_train[num_cols].median()
    df_train[num_cols] = df_train[num_cols].fillna(train_medians)
    df_test[num_cols]  = df_test[num_cols].fillna(train_medians)

    return df_train, df_test, train_medians


# Обучение всех моделей

@st.cache_resource
def train_all_models(_df_train, _df_test):
    df_train = _df_train
    df_test  = _df_test
    target   = "selling_price"

    # Задание 11: числовые признаки
    num_features = ["year", "km_driven", "mileage", "engine",
                    "max_power", "torque", "max_torque_rpm"]

    X_train_num = df_train[num_features]
    X_test_num  = df_test[num_features]
    y_train     = df_train[target]
    y_test      = df_test[target]

    # Задание 14: стандартизация — fit только на train
    scaler_num     = StandardScaler()
    X_train_scaled = scaler_num.fit_transform(X_train_num)
    X_test_scaled  = scaler_num.transform(X_test_num)

    # Задание 12: Линейная регрессия
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_train_lr = lr.predict(X_train_scaled)
    y_pred_test_lr  = lr.predict(X_test_scaled)

    # Задание 17: Lasso + GridSearchCV 10 фолдов
    lasso_gs = GridSearchCV(
        Lasso(random_state=RANDOM_STATE, max_iter=10000),
        param_grid={"alpha": [0.01, 0.1, 1, 10, 100, 1000, 5000, 10000, 50000, 100000]},
        cv=10, scoring="r2", n_jobs=-1
    )
    lasso_gs.fit(X_train_scaled, y_train)
    best_lasso         = lasso_gs.best_estimator_
    y_pred_train_lasso = best_lasso.predict(X_train_scaled)
    y_pred_test_lasso  = best_lasso.predict(X_test_scaled)

    # Задание 17: ElasticNet + GridSearchCV
    en_gs = GridSearchCV(
        ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
        param_grid={
            "alpha":    [0.01, 0.1, 1, 10, 100, 1000, 5000, 10000, 50000, 100000],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        },
        cv=10, scoring="r2", n_jobs=-1
    )
    en_gs.fit(X_train_scaled, y_train)
    best_en         = en_gs.best_estimator_
    y_pred_train_en = best_en.predict(X_train_scaled)
    y_pred_test_en  = best_en.predict(X_test_scaled)

    # Задания 19–22: Ridge + OHE + GridSearchCV
    cat_features_ohe = ["fuel", "seller_type", "transmission", "owner", "brand", "seats"]
    X_train_cat = pd.concat([X_train_num, df_train[cat_features_ohe]], axis=1)
    X_test_cat  = pd.concat([X_test_num,  df_test[cat_features_ohe]],  axis=1)

    # Задание 20: OHE кодирование
    X_train_ohe = pd.get_dummies(X_train_cat, columns=cat_features_ohe, drop_first=True)
    X_test_ohe  = pd.get_dummies(X_test_cat,  columns=cat_features_ohe, drop_first=True)
    X_test_ohe  = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)
    ohe_columns = X_train_ohe.columns.tolist()

    scaler_cat         = StandardScaler()
    X_train_ohe_scaled = scaler_cat.fit_transform(X_train_ohe)
    X_test_ohe_scaled  = scaler_cat.transform(X_test_ohe)

    ridge_gs = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [0.01, 0.1, 1, 10, 100, 1000, 5000, 10000, 50000, 100000]},
        cv=10, scoring="r2", n_jobs=-1
    )
    ridge_gs.fit(X_train_ohe_scaled, y_train)
    best_ridge         = ridge_gs.best_estimator_
    y_pred_train_ridge = best_ridge.predict(X_train_ohe_scaled)
    y_pred_test_ridge  = best_ridge.predict(X_test_ohe_scaled)

    # Сводка метрик
    metrics = {
        "LinearRegression (num)": get_metrics(
            y_train, y_pred_train_lr, y_test, y_pred_test_lr),
        f"Lasso (α={lasso_gs.best_params_['alpha']})": get_metrics(
            y_train, y_pred_train_lasso, y_test, y_pred_test_lasso),
        f"ElasticNet (α={en_gs.best_params_['alpha']}, l1={en_gs.best_params_['l1_ratio']})": get_metrics(
            y_train, y_pred_train_en, y_test, y_pred_test_en),
        f"Ridge OHE (α={ridge_gs.best_params_['alpha']})": get_metrics(
            y_train, y_pred_train_ridge, y_test, y_pred_test_ridge),
    }

    return {
        "y_train": y_train, "y_test": y_test,
        "lr": lr, "scaler_num": scaler_num,
        "y_pred_train_lr": y_pred_train_lr, "y_pred_test_lr": y_pred_test_lr,
        "lasso": best_lasso, "lasso_alpha": lasso_gs.best_params_["alpha"],
        "y_pred_train_lasso": y_pred_train_lasso, "y_pred_test_lasso": y_pred_test_lasso,
        "en": best_en, "en_params": en_gs.best_params_,
        "y_pred_train_en": y_pred_train_en, "y_pred_test_en": y_pred_test_en,
        "ridge": best_ridge, "scaler_cat": scaler_cat,
        "ohe_columns": ohe_columns, "ridge_alpha": ridge_gs.best_params_["alpha"],
        "y_pred_train_ridge": y_pred_train_ridge, "y_pred_test_ridge": y_pred_test_ridge,
        "num_features": num_features,
        "cat_features_ohe": cat_features_ohe,
        "metrics": metrics,
    }


# Предсказание

def predict_one(row_dict, results):
    num_features     = results["num_features"]
    cat_features_ohe = results["cat_features_ohe"]
    ohe_columns      = results["ohe_columns"]
    scaler_cat       = results["scaler_cat"]
    ridge            = results["ridge"]

    df    = pd.DataFrame([row_dict])
    X     = df[num_features + cat_features_ohe]
    X_ohe = pd.get_dummies(X, columns=cat_features_ohe, drop_first=True)
    X_ohe = X_ohe.reindex(columns=ohe_columns, fill_value=0)
    return max(0, ridge.predict(scaler_cat.transform(X_ohe))[0])


def predict_batch_df(df_input, results, medians):
    df = df_input.copy()
    if "mileage" in df.columns and df["mileage"].dtype == object:
        df = preprocess_features(df)
    if "brand" not in df.columns:
        df = add_brand(df)

    num_cols = [c for c in medians.index if c in df.columns]
    df[num_cols] = df[num_cols].fillna(medians[num_cols])

    num_features     = results["num_features"]
    cat_features_ohe = results["cat_features_ohe"]
    ohe_columns      = results["ohe_columns"]

    avail_cat = [c for c in cat_features_ohe if c in df.columns]
    avail_num = [c for c in num_features if c in df.columns]
    X     = df[avail_num + avail_cat]
    X_ohe = pd.get_dummies(X, columns=avail_cat, drop_first=True)
    X_ohe = X_ohe.reindex(columns=ohe_columns, fill_value=0)
    return np.maximum(results["ridge"].predict(results["scaler_cat"].transform(X_ohe)), 0)


# ИНТЕРФЕЙС

st.title("Car Price Predictor")
st.markdown("*Предсказание стоимости подержанных автомобилей*")

with st.spinner("Загружаем данные и обучаем модели..."):
    df_train_raw, df_test_raw = load_data()
    df_train, df_test, medians = full_pipeline(df_train_raw, df_test_raw)
    results = train_all_models(df_train, df_test)

st.success(f"✓ Train: {len(df_train):,} объектов | Test: {len(df_test):,} объектов")

tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Метрики моделей", "Предсказание", "Веса модели"])


# TAB 1 — EDA 

with tab1:
    st.header("Exploratory Data Analysis")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Объектов в train", "Медианная цена", "Уникальных марок", "Годы выпуска"],
        [
            f"{len(df_train):,}",
            f"₽{int(df_train['selling_price'].median()):,}",
            str(df_train["brand"].nunique()),
            f"{df_train['year'].min()}–{df_train['year'].max()}"
        ]
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Распределение цены
    st.subheader("Распределение целевой переменной (selling_price)")
    col1, col2 = st.columns(2)

    with col1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(df_train["selling_price"], bins=50,
                     color="#2563eb", edgecolor="white", alpha=0.85)
        axes[0].set_title("Оригинал"); axes[0].set_xlabel("Цена (₽)")
        axes[1].hist(np.log1p(df_train["selling_price"]), bins=50,
                     color="#f59e0b", edgecolor="white", alpha=0.85)
        axes[1].set_title("log(1 + x)"); axes[1].set_xlabel("log(Цена)")
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        fuel_order = (df_train.groupby("fuel")["selling_price"]
                      .median().sort_values(ascending=False).index)
        sns.boxplot(data=df_train, x="fuel", y="selling_price",
                    order=fuel_order, palette="Set2", ax=ax)
        ax.set_title("Цена по типу топлива")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        stats = (df_train.groupby("transmission")["selling_price"]
                 .median().sort_values(ascending=False))
        bars = ax.bar(stats.index, stats.values,
                      color=["#2563eb", "#f59e0b"], edgecolor="white", width=0.5)
        for bar, val in zip(bars, stats.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 5000, f"₽{val:,.0f}", ha="center", fontsize=9)
        ax.set_title("Медианная цена по КПП")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        owner_order = ["First Owner", "Second Owner", "Third Owner",
                       "Fourth & Above Owner", "Test Drive Car"]
        order_valid = [o for o in owner_order if o in df_train["owner"].unique()]
        sns.boxplot(data=df_train, x="owner", y="selling_price",
                    order=order_valid, palette="coolwarm", ax=ax)
        ax.set_title("Цена по числу владельцев")
        ax.tick_params(axis="x", rotation=20)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    #  Тепловая карта корреляций Пирсона
    st.subheader("Матрица корреляций Пирсона")
    num_cols_viz = ["selling_price", "year", "km_driven", "mileage",
                    "engine", "max_power", "torque", "seats", "max_torque_rpm"]
    corr_matrix = df_train[num_cols_viz].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Корреляционная матрица (Пирсон) — Train")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Возраст vs цена
    st.subheader("Возраст автомобиля vs Цена")
    df_viz = df_train.copy()
    df_viz["car_age"] = 2024 - df_viz["year"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df_viz["car_age"], df_viz["selling_price"],
               alpha=0.2, s=10, color="steelblue")
    log_price = np.log1p(df_viz["selling_price"])
    z = np.polyfit(df_viz["car_age"], log_price, 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_viz["car_age"].min(), df_viz["car_age"].max(), 100)
    ax.plot(x_line, np.expm1(p(x_line)), color="red", linewidth=2, label="тренд")
    ax.set_xlabel("Возраст (лет)"); ax.set_ylabel("Цена (₽)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# TAB 2 — Метрики 

with tab2:
    st.header("Сравнение всех моделей")

    metrics_df = pd.DataFrame(results["metrics"]).T
    metrics_df.index.name = "Модель"

    st.subheader("R² и RMSE ")
    st.dataframe(
        metrics_df[["R² Train", "R² Test", "RMSE Train", "RMSE Test"]]
        .style
        .format({"R² Train": "{:.4f}", "R² Test": "{:.4f}",
                 "RMSE Train": "{:,}", "RMSE Test": "{:,}"})
        .background_gradient(subset=["R² Test"], cmap="RdYlGn"),
        use_container_width=True
    )

    st.subheader("Бизнес-метрики")
    st.dataframe(
        metrics_df[["Business Train", "Business Test", "Asymmetric Train", "Asymmetric Test"]]
        .style.format("{:.4f}")
        .background_gradient(subset=["Business Test"], cmap="RdYlGn")
        .background_gradient(subset=["Asymmetric Test"], cmap="RdYlGn_r"),
        use_container_width=True
    )
    st.caption("**Business** — доля ±10% ↑ лучше | **Asymmetric** — недопрогноз ×2 штраф ↓ лучше")

    # Scatter: предсказанные vs реальные (Ridge)
    st.subheader("Ridge: предсказанные vs реальные цены")
    y_test       = results["y_test"]
    y_pred_ridge = results["y_pred_test_ridge"]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred_ridge, alpha=0.3, s=10, color="steelblue")
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], color="red", linewidth=2, label="идеал")
        ax.fill_between([y_test.min(), y_test.max()],
                        [y_test.min()*0.9, y_test.max()*0.9],
                        [y_test.min()*1.1, y_test.max()*1.1],
                        alpha=0.2, color="green", label="±10%")
        ax.set_xlabel("Реальная цена"); ax.set_ylabel("Предсказанная цена")
        ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        rel_err = (y_pred_ridge - y_test) / y_test
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(rel_err[rel_err < 0], bins=40, color="coral",
                alpha=0.7, label="Недопрогноз (штраф ×2)")
        ax.hist(rel_err[rel_err >= 0], bins=40, color="steelblue",
                alpha=0.7, label="Перепрогноз")
        ax.axvline(0, color="black", linewidth=1.5)
        ax.axvline(-0.1, color="red", linewidth=1, linestyle="--", label="±10%")
        ax.axvline(0.1,  color="red", linewidth=1, linestyle="--")
        ax.set_xlabel("Относительная ошибка")
        ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Коэффициенты Lasso 
    st.subheader(f"Коэффициенты Lasso α={results['lasso_alpha']}")
    lasso_coef = pd.DataFrame({
        "feature":     results["num_features"],
        "coefficient": results["lasso"].coef_
    }).sort_values("coefficient", key=abs, ascending=True)

    n_zero = (lasso_coef["coefficient"] == 0).sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["coral" if c < 0 else "steelblue" for c in lasso_coef["coefficient"]]
    ax.barh(lasso_coef["feature"], lasso_coef["coefficient"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Lasso: зануленных признаков = {n_zero}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()



# TAB 3 — Предсказание

with tab3:
    st.header("Предсказание цены автомобиля")
    st.caption(f"Используется Ridge с OHE (α={results['ridge_alpha']})")

    mode = st.radio("Режим:", ["Ввести вручную", "Загрузить CSV"], horizontal=True)

    if mode == "Ввести вручную":
        col1, col2, col3 = st.columns(3)

        with col1:
            brand        = st.selectbox("Марка", sorted(df_train["brand"].unique()))
            year         = st.slider("Год выпуска", 1990, 2020, 2015)
            km_driven    = st.number_input("Пробег (км)", 0, 500000, 50000, step=1000)
            fuel         = st.selectbox("Тип топлива", df_train["fuel"].unique())

        with col2:
            transmission = st.selectbox("КПП", df_train["transmission"].unique())
            owner        = st.selectbox("Владелец", df_train["owner"].unique())
            seller_type  = st.selectbox("Тип продавца", df_train["seller_type"].unique())
            seats        = st.selectbox("Мест", sorted(df_train["seats"].unique()))

        with col3:
            mileage        = st.number_input("Расход (kmpl)", 0.0, 50.0, 18.0, step=0.1)
            engine         = st.number_input("Объём (CC)", 500, 5000, 1200, step=50)
            max_power      = st.number_input("Мощность (bhp)", 0.0, 500.0, 80.0, step=1.0)
            torque         = st.number_input("Момент (Nm)", 0.0, 800.0, 150.0, step=5.0)
            max_torque_rpm = st.number_input("Обороты макс. момента", 500, 10000, 3000, step=100)

        if st.button("Рассчитать цену", type="primary", use_container_width=True):
            row = {
                "brand": brand, "year": year, "km_driven": km_driven,
                "fuel": fuel, "transmission": transmission, "owner": owner,
                "seller_type": seller_type, "seats": float(seats),
                "mileage": mileage, "engine": float(engine), "max_power": max_power,
                "torque": torque, "max_torque_rpm": float(max_torque_rpm)
            }
            price = predict_one(row, results)
            st.markdown(f"""
            <div class="prediction-box">
                <div class="metric-label" style="color:rgba(255,255,255,0.7)">Предсказанная стоимость</div>
                <div class="prediction-price">₽ {price:,.0f}</div>
                <div style="font-size:0.9rem;opacity:0.8">{brand} {year} · {fuel} · {transmission}</div>
            </div>""", unsafe_allow_html=True)

            similar = df_train[
                (df_train["brand"] == brand) &
                (df_train["fuel"] == fuel) &
                (df_train["year"].between(year - 2, year + 2))
            ][["brand", "year", "km_driven", "fuel", "transmission", "selling_price"]].head(5)

            st.subheader("Похожие автомобили в обучающей выборке")
            if len(similar) > 0:
                st.dataframe(similar.style.format(
                    {"selling_price": "₽{:,.0f}", "km_driven": "{:,}"}),
                    use_container_width=True)
            else:
                st.info("Похожих автомобилей не найдено")

    else:
        st.caption("Ожидаемые столбцы: name, year, km_driven, fuel, seller_type, "
                   "transmission, owner, mileage, engine, max_power, torque, seats")
        uploaded = st.file_uploader("Выберите CSV", type="csv")

        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up.head(), use_container_width=True)

            if st.button("Предсказать", type="primary"):
                preds  = predict_batch_df(df_up, results, medians)
                df_res = df_up.copy()
                df_res["predicted_price"] = preds.round(0).astype(int)
                st.success(f"Готово! Предсказано {len(df_res)} объектов")
                show = (["name", "year", "km_driven", "fuel", "transmission", "predicted_price"]
                        if "name" in df_res.columns else df_res.columns.tolist())
                st.dataframe(
                    df_res[show].style.format({"predicted_price": "₽{:,}"}),
                    use_container_width=True)
                st.download_button("⬇Скачать результаты",
                                   df_res.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")


# TAB 4 — Веса модели 

with tab4:
    st.header(f"Веса Ridge (α={results['ridge_alpha']}, num + OHE)")

    coef_df = pd.DataFrame({
        "feature":     results["ohe_columns"],
        "coefficient": results["ridge"].coef_
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        n_show = st.slider("Топ-N признаков", 10, min(50, len(coef_df)), 20)
        top_n  = coef_df.head(n_show).sort_values("coefficient")

        fig, ax = plt.subplots(figsize=(9, max(5, n_show * 0.35)))
        colors = ["#ef4444" if c < 0 else "#2563eb" for c in top_n["coefficient"]]
        ax.barh(top_n["feature"], top_n["coefficient"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Коэффициент β")
        ax.set_title(f"Топ-{n_show} признаков по важности")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Таблица весов")
        st.dataframe(
            coef_df[["feature", "coefficient"]].head(n_show)
            .style.format({"coefficient": "{:,.0f}"})
            .background_gradient(subset=["coefficient"], cmap="RdBu",
                                  vmin=-coef_df["abs_coef"].max(),
                                  vmax=coef_df["abs_coef"].max()),
            use_container_width=True, height=500
        )

    # Суммарная важность по группам
    st.subheader("Суммарная важность по группам признаков")

    def get_group(feat):
        for g in ["fuel", "transmission", "owner", "seller_type", "brand", "seats",
                  "year", "km_driven", "mileage", "engine",
                  "max_power", "torque", "max_torque_rpm"]:
            if feat.startswith(g):
                return g
        return feat

    coef_df["group"] = coef_df["feature"].apply(get_group)
    group_imp = coef_df.groupby("group")["abs_coef"].sum().sort_values(ascending=False)

    num_groups = {"year", "km_driven", "mileage", "engine",
                  "max_power", "torque", "max_torque_rpm"}
    colors = ["#2563eb" if g in num_groups else "#f59e0b" for g in group_imp.index]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(group_imp.index, group_imp.values, color=colors, edgecolor="white", alpha=0.85)
    ax.set_ylabel("Суммарный |β|")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.caption("🔵 Числовые признаки  🟡 Категориальные признаки (OHE)")
