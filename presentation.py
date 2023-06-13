import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/train.tsv", sep="\t")

st.markdown("# NUM Project ")
st.write("Tools that were used:")
st.write("DVC, MLFlow, Streamlit")
if st.sidebar.checkbox('Show chart'):
    st.write("Amount of every sentiment in dataframe")
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    chart_data = data["Sentiment"].value_counts()
    ax.bar(data["Sentiment"].value_counts().index, data["Sentiment"].value_counts().values)
    st.pyplot(fig)


st.write(data)
