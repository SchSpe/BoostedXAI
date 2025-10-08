import streamlit as st
import pandas as pd
import numpy as np
from anchor import anchor_tabular

st.header("BoostedXAI: :blue[Anchors] and :orange[Counterfactuals]", divider=True)

with st.container(border=True):
    datafile = st.file_uploader("Upload CSV", type=["csv"])

if datafile:
    data = pd.read_csv(datafile)    
    features = data.keys().to_list()

    # Set Target Feature
    with st.container(border=True):
        target = st.segmented_control("Target Feature:", options=features, selection_mode="single", key="target_feature")
        st.markdown(f"Target Feature: {target}")
        if target: 
            class_names = data[target].unique().tolist()
            st.markdown(f"Class Names: {class_names}")

            df = pd.DataFrame({"Class": class_names, "Name": ["Change Me!"] * len(class_names)})

            class_names = st.data_editor(
                df,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "Field": st.column_config.Column(disabled=True),
                    "Value": st.column_config.TextColumn("Write Class Names Here")})
            #st.write(class_names)
            class_names = class_names['Name'].tolist()
            st.markdown(f"Class Names: {class_names}")


    # Set Categorical Features
    with st.container(border=True):
        categorical_features = st.segmented_control("Select Categorical Features", options=features, selection_mode="multi")
        st.markdown(f"Categorical Features: {categorical_features}")

        