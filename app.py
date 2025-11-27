import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------
# LOAD & PREPARE DATA
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("autos.csv.bz2", encoding="iso-8859-1")
    df = df[df["offerType"] == "Angebot"]
    df = df[df["vehicleType"] == "kleinwagen"]
    df = df[df["notRepairedDamage"] == "nein"]
    df.dropna(inplace=True)
    return df

df = load_data()

# Feature columns
X = df[["kilometer", "yearOfRegistration", "brand"]]
y = df["price"]

# Encoder
cf = ColumnTransformer([
    ("brand", OneHotEncoder(drop="first"), ["brand"])
], remainder="passthrough")

X_transformed = cf.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, train_size=0.75, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("🚗 Predikcija cijene auta – AutoScout Model")

st.write("Unesi podatke o vozilu i model će procijeniti cijenu.")

# INPUTS
kilometer = st.number_input("Kilometraža", min_value=0, max_value=400000, value=150000)
year = st.number_input("Godina registracije", min_value=1950, max_value=2025, value=2000)

brand = st.selectbox("Marka vozila", sorted(df["brand"].unique()))

# Predict button
if st.button("🔮 Izračunaj cijenu"):
    X_pred = pd.DataFrame([[kilometer, year, brand]],
                          columns=["kilometer", "yearOfRegistration", "brand"])

    pred = model.predict(cf.transform(X_pred))[0]

    st.success(f"Procijenjena cijena: **{int(pred):,} €**")

# Display model scores
st.sidebar.header("Model evaluacija")
st.sidebar.write(f"R2 (train): {model.score(X_train, y_train):.3f}")
st.sidebar.write(f"R2 (test): {model.score(X_test, y_test):.3f}")
