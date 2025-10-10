import streamlit as st
import pandas as pd
import numpy as np
from anchor import anchor_tabular
import dill
import os

st.header("BoostedXAI: :blue[Anchors] and :orange[Counterfactuals]", divider=True)

with st.container(border=True):
    datafile = st.file_uploader("Upload CSV", type=["csv"])
    modelfile = st.file_uploader("Upload Model", type=["pkl", "joblib", "modelfile"])

if datafile and modelfile:
    data = pd.read_csv(datafile)
    st.session_state["data"] = data 
    features = data.keys().to_list()

    # Getting model from uploaded file
    if "temp" not in os.listdir():
        os.mkdir("./temp")

    with open("./temp/temp_model.modelfile", "wb") as f:
        f.write(modelfile.getvalue())
    with open("./temp/temp_model.modelfile", "rb") as f:
        model = dill.load(f)

    r, c = st.session_state["data"].shape
            
    st.session_state["model"] = model

    # Set Target Feature
    with st.container(border=True):
        target = st.segmented_control("Target Feature:", options=features, selection_mode="single", key="target_feature")
        st.markdown(f"Target Feature: {target}")
        if target:
            # Default class labels = original target values (reset when target changes)
            orig_classes = pd.Series(data[target]).dropna().unique().tolist()
            if st.session_state.get("target_feature_prev") != target:
                st.session_state["class_names"] = [str(x) for x in orig_classes]
                st.session_state["edit_classes_open"] = False
                st.session_state["target_feature_prev"] = target

            st.markdown(f"Class Names: {st.session_state['class_names']}")

            # Button to toggle editing of target class labels
            btn_label = "Edit target class labels" if not st.session_state.get("edit_classes_open", False) else "Done editing"
            if st.button(btn_label, key="toggle_edit_classes"):
                st.session_state["edit_classes_open"] = not st.session_state.get("edit_classes_open", False)
                st.rerun()

            # Editable table only when editing is toggled on; defaults to original values
            if st.session_state.get("edit_classes_open", False):
                df = pd.DataFrame({"Class": orig_classes, "Name": st.session_state["class_names"]})
                edited_df = st.data_editor(
                    df,
                    hide_index=True,
                    num_rows="fixed",
                    column_config={
                        "Class": st.column_config.Column(disabled=True),
                        "Name": st.column_config.TextColumn("New Class Name"),
                    })
                st.session_state["class_names"] = edited_df['Name'].tolist()

            # Remove target from feature list used elsewhere
            features.remove(target)


    # Set Categorical Features
    with st.container(border=True):
        categorical_features = st.segmented_control("Select Categorical Features", options=features, selection_mode="multi")
        st.markdown(f"Categorical Features: {categorical_features}")

        # String/object columns are automatically treated as categorical
        auto_cats = [c for c in features if data[c].dtype == object or getattr(data[c].dtype, 'name', '') == 'object']
        selected_cats = sorted(set(categorical_features).union(auto_cats), key=lambda n: features.index(n))

        # Automatic inference of categorical labels
        name_to_idx = {c: i for i, c in enumerate(features)}
        categorical_names_map = {
            name_to_idx[c]: pd.Series(data[c]).dropna().unique().astype(str).tolist()
            for c in selected_cats if c in name_to_idx
        }
        st.session_state["categorical_features"] = [
            name_to_idx[c] for c in selected_cats if c in name_to_idx
        ]
        st.session_state["categorical_names"] = categorical_names_map

        # Encode selected categorical columns to integer codes
        X = data[features].copy()
        for c in selected_cats:
            idx = name_to_idx[c]
            order = st.session_state["categorical_names"][idx]
            X[c] = pd.Categorical(X[c].astype(str), categories=order).codes
        st.session_state["data"] = X.values

        # Preview of inferred categorical labels
        idx_to_name = {i: c for c, i in name_to_idx.items()}
        preview_rows = []
        for idx in sorted(categorical_names_map.keys()):
            labels = categorical_names_map[idx]
            preview_rows.append({
                "Feature": f"{idx_to_name.get(idx, 'Unknown')} (Column: {idx})",
                "Labels": ", ".join(labels) if labels else "—",
            })
        if preview_rows:
            st.subheader("Inferred categorical labels")
            st.dataframe(pd.DataFrame(preview_rows), hide_index=True)
        else:
            st.info("No categorical features selected.")

# Initialize Anchor Explainer when prerequisites are met
if "data" in st.session_state and "model" in st.session_state and "target_feature" in st.session_state:
    train = st.session_state["data"]

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=st.session_state.get("class_names", None),
        feature_names=features,
        train_data=train,
        categorical_names=st.session_state.get("categorical_names", None),
    )

    # Choose a row to explain, or enter a custom row
    with st.container(border=True):
        use_custom = st.checkbox("Enter a custom row to explain", value=False)
        if use_custom:
            df_one = pd.DataFrame([train[0]], columns=features)
            edited = st.data_editor(df_one, hide_index=True, num_rows="fixed")
            row = edited.iloc[0].to_numpy()
        else:
            n_rows = int(train.shape[0])
            i_row = st.number_input("Row index to explain", min_value=0, max_value=max(0, n_rows-1), value=0, step=1)
            row = train[int(i_row)]
            st.markdown("**Selected row**")
            st.dataframe(pd.DataFrame([row], columns=features), hide_index=True)

        exp = explainer.explain_instance(row, model.predict, threshold=0.95)
        with st.container(border=True):
            st.subheader("Anchor Explanation")
            st.write('Anchor: %s' % (' AND '.join(exp.names())))
            st.write('Precision: %.2f' % exp.precision())
            st.write('Coverage: %.2f' % exp.coverage())