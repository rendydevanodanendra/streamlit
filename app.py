import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# --- Judul Aplikasi ---
st.title("Klasifikasi Data dengan Decision Tree")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload file CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Awal:")
    st.dataframe(df)

    # --- Preprocessing ---
    st.subheader("Preprocessing")
    target_col = st.selectbox("Pilih kolom target", df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Imputasi jika ada NaN
        imp = SimpleImputer(strategy="mean")
        X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

        # Scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns)

        # SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_scaled, y)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # --- Training ---
        st.subheader("Training Decision Tree")
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # --- Evaluasi ---
        st.subheader("Evaluasi")
        acc = clf.score(X_test, y_test)
        st.write(f"Akurasi: {acc:.2f}")
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.write("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # ROC Curve (jika binary)
        if len(np.unique(y)) == 2:
            y_prob = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            st.write("ROC Curve")
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax2.plot([0, 1], [0, 1], "k--")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.legend()
            st.pyplot(fig2)
