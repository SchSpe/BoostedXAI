import streamlit as st
import pandas as pd
import numpy as np
from anchor import anchor_tabular
import dill
import os
import dice_ml
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif


def train_mlp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return model, acc

def symmetrical_uncertainty(X, y):
    mi = mutual_info_classif(X, y, discrete_features='auto')
    # approximate entropy for each feature
    Hx = np.array([np.log2(len(np.unique(X[:, i]))) for i in range(X.shape[1])])
    Hy = np.log2(len(np.unique(y)))
    return 2 * mi / (Hx + Hy)

def fcbf(X, y, threshold=0.01):
    su = symmetrical_uncertainty(X, y)
    ranked = sorted(list(enumerate(su)), key=lambda x: x[1], reverse=True)
    selected = [i for i, score in ranked if score >= threshold]
    return selected, ranked


def necessity_test(exp_dice, instance_df, desired_class, feature):
    # A feature value xi is necessary for causing the model output y if changing xi changes the model output, while keeping every other feature constant.
    # If features_to_vary is set to , then the generated counterfactuals demonstrate the necessity of the feature
    vary_list = [feature]
    cf = exp_dice.generate_counterfactuals(
        instance_df,
        total_CFs=20,
        desired_class=desired_class,
        features_to_vary=vary_list,
    )
    return True

def sufficiency_test(exp_dice, instance_df, desired_class, feature):
    # A feature value  is sufficient for causing the model output  if it is impossible to change the model output while keeping  constant.
    # If features_to_vary is set to all features except , then the absence of any counterfactuals demonstrates the sufficiency of the feature.
    all_cols = list(instance_df.columns)
    vary_list = [c for c in all_cols if c != feature]
    cf = exp_dice.generate_counterfactuals(
        instance_df,
        total_CFs=50,
        desired_class=desired_class,
        features_to_vary=vary_list
    )
    return False


def nec_and_suf_ui(exp_dice, instance_df, desired_class, anchor_features):
    st.markdown("### 🔍 Necessity and Sufficiency Tests")
    st.caption("Test if anchor features are necessary or sufficient for the model's prediction.")

    for feature in anchor_features:
        try:
            is_necessary = necessity_test(exp_dice, instance_df, desired_class, feature)
        except Exception as e:
            is_necessary = False
        try:
            is_sufficient = sufficiency_test(exp_dice, instance_df, desired_class, feature)
        except Exception as e:
            is_sufficient = False

        col1, col2 = st.columns(2)
        with col1:
            emoji_n = "✅" if is_necessary else "❌"
            st.write(f"{emoji_n} **{feature} is{' ' if is_necessary else ' not '}necessary**")
            st.caption("Changing only this feature can change the prediction." if is_necessary else "Changing only this feature does not change the prediction.")
        with col2:
            emoji_s = "✅" if is_sufficient else "❌"
            st.write(f"{emoji_s} **{feature} is{' ' if is_sufficient else ' not '}sufficient**")
            st.caption("Keeping only this feature constant keeps the prediction the same." if is_sufficient else "Keeping only this feature constant does not guarantee the same prediction.")


st.header("BoostedXAI: :blue[Anchors] and :orange[Counterfactuals]", divider=True)

with st.container(border=True):
    datafile = st.file_uploader("Upload CSV", type=["csv"])
    custom = st.toggle("Use custom model", value=False)
    st.caption("If not using a custom model, a default MLP classifier will be trained on the uploaded data.")
    if custom:
        modelfile = st.file_uploader("Upload Model", type=["joblib"])

        if modelfile:
            backend_choice = st.segmented_control(
                "Select model type for DiCE",
                options=["sklearn", "TF2", "PYT", "TF1"],
                selection_mode="single",
                key="dice_backend"
            )

            # Getting model from uploaded file
            if "temp" not in os.listdir():
                os.mkdir("./temp")

            with open("./temp/temp_model.modelfile", "wb") as f:
                f.write(modelfile.getvalue())
            with open("./temp/temp_model.modelfile", "rb") as f:
                model = dill.load(f)

            st.session_state["model"] = model


if datafile:
    data = pd.read_csv(datafile)
    st.session_state["raw_data"] = data
    features = data.keys().to_list()

    # Set Target Feature
    with st.container(border=True):
        target = st.segmented_control("Target Feature:", options=features, selection_mode="single", key="target_feature")
        st.markdown(f"Target Feature: {target}")
        if target:
            st.session_state["y_true"] = data[target]
            # Default class labels
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

            # Editable table only when editing is toggled on
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

            # Remove target from feature list
            features.remove(target)

    # Set Categorical Features
    with st.container(border=True):
        categorical_features = st.segmented_control("Select Categorical Features", options=features, selection_mode="multi")
        st.markdown(f"Categorical Features: {categorical_features}")

        # Auto-detect object columns as categorical
        auto_cats = [c for c in features if data[c].dtype == object or getattr(data[c].dtype, 'name', '') == 'object']
        selected_cats = sorted(set(categorical_features).union(auto_cats), key=lambda n: features.index(n))

        # Map categorical names
        name_to_idx = {c: i for i, c in enumerate(features)}
        categorical_names_map = {
            name_to_idx[c]: pd.Series(data[c]).dropna().unique().astype(str).tolist()
            for c in selected_cats if c in name_to_idx
        }
        st.session_state["categorical_features"] = [
            name_to_idx[c] for c in selected_cats if c in name_to_idx
        ]
        st.session_state["categorical_names"] = categorical_names_map

        # Encode categorical features
        X = data[features].copy()
        for c in selected_cats:
            idx = name_to_idx[c]
            order = st.session_state["categorical_names"][idx]
            X[c] = pd.Categorical(X[c].astype(str), categories=order).codes

        st.session_state["encoded_data"] = X
        st.session_state["data_matrix"] = X.values

        # FCBF feature selection
        apply_fcbf = st.toggle("Apply FCBF Feature Selection", value=False)

        if apply_fcbf:
            st.subheader("FCBF Feature Ranking")

            X_mat = X.values
            y_vec = data[target].values

            selected_idx, ranking = fcbf(X_mat, y_vec)

            rank_df = pd.DataFrame({
                "Feature": [features[i] for i, _ in ranking],
                "Score": [round(s, 4) for _, s in ranking],
                "Selected": ["Yes" if i in selected_idx else "No" for i, _ in ranking]
            })

            st.dataframe(rank_df, hide_index=True)

            if len(selected_idx) > 0:
                selected_features = [features[i] for i in selected_idx]
                st.success(f"Using {len(selected_features)} selected features")

                X = X[selected_features].copy()
                features = selected_features
                st.session_state["encoded_data"] = X
                st.session_state["data_matrix"] = X.values
                st.session_state.pop("model", None)
            else:
                st.warning("No features passed the threshold. Keeping all features.")

        # Preview
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

    # Train model if not using custom model, only once per session
    if target and not custom and "model" not in st.session_state:
        X_encoded = st.session_state["encoded_data"].copy()
        y_encoded = data[target]
        with st.spinner("Training MLP..."):
            model, acc = train_mlp(X_encoded, y_encoded)
        st.session_state["model"] = model
        st.session_state["acc"] = acc
        st.success(f"Model trained with accuracy: {acc:.2%}")
    elif target and not custom:
        st.info(f"Using existing model (accuracy: {st.session_state.get('acc', float('nan')):.2%})")

# Main Explanation Section
if "data_matrix" in st.session_state and "model" in st.session_state and "target_feature" in st.session_state:
    train = st.session_state["data_matrix"]

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=st.session_state.get("class_names", None),
        feature_names=features,
        train_data=train,
        categorical_names=st.session_state.get("categorical_names", None),
    )

    # Choose row
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

        col1, col2 = st.columns(2)
        # Anchor explanation
        with col1:
            exp = explainer.explain_instance(row, st.session_state["model"].predict, threshold=0.95)
            with st.container(border=True):
                st.subheader("Anchor Explanation")
                st.write('Anchor: %s' % (' AND '.join(exp.names())))
                st.write('Precision: %.2f' % exp.precision())
                st.write('Coverage: %.2f' % exp.coverage())
                anchor_conditions = exp.names()
                anchor_features = {f for f in features for cond in anchor_conditions if f in cond in anchor_conditions}
        # Counterfactual explanation
        with col2:
            with st.container(border=True):
                st.subheader("Counterfactual Explanation")
                try:
                    with st.spinner("Generating counterfactuals..."):
                        X_train_df = pd.DataFrame(train, columns=features)
                        y_train = pd.Series(st.session_state.get("y_true"))
                        train_df = X_train_df.copy()
                        train_df[target] = y_train.values

                        cat_idx = st.session_state.get("categorical_features", [])
                        cat_names = [features[i] for i in cat_idx if i < len(features)]
                        cont_features = [c for c in X_train_df.columns if c not in cat_names]

                        data_dice = dice_ml.Data(
                            dataframe=train_df,
                            continuous_features=cont_features,
                            categorical_features=cat_names,
                            outcome_name=target
                        )
                        backend = st.session_state.get("dice_backend", "sklearn") if custom else "sklearn"
                        model_dice = dice_ml.Model(model=st.session_state["model"], backend=backend)
                        exp_dice = dice_ml.Dice(data_dice, model_dice, method='random')

                        instance_df = pd.DataFrame([row], columns=features)

                        # Derive labels directly from the target column to avoid type mismatches
                        labels = pd.Series(train_df[target]).dropna().unique().tolist()

                        if len(labels) == 2:
                            desired_class = "opposite"
                        else:
                            current_pred = st.session_state["model"].predict(instance_df)[0]
                            # Pick any label different from the current prediction (guaranteed same dtype)
                            desired_class = next((lbl for lbl in labels if lbl != current_pred), labels[0])

                        cf = exp_dice.generate_counterfactuals(
                            instance_df,
                            total_CFs=3,
                            desired_class=("opposite" if len(labels)==2 else int(list((st.session_state["model"].steps[-1][1].classes_ if hasattr(st.session_state["model"], "steps") else st.session_state["model"].classes_)).index(desired_class)))
                        )

                        cf_df = cf.cf_examples_list[0].final_cfs_df if hasattr(cf, 'cf_examples_list') else None
                        if cf_df is not None and len(cf_df) > 0:
                            anchor_conditions = exp.names()
                            anchor_features = {f for f in features for cond in anchor_conditions if f in cond}
                            for cf_idx in range(len(cf_df)):
                                changes = []
                                for col in X_train_df.columns:
                                    original = float(instance_df.iloc[0][col])
                                    cf_val = float(cf_df.iloc[cf_idx][col])
                                    if abs(original - cf_val) > 1e-6:
                                        in_anchor = col in anchor_features
                                        emoji = "❌" if in_anchor else "✅"
                                        changes.append(f"{emoji} {col}: {original:.2f} → {cf_val:.2f}")
                                if changes:
                                    st.markdown(f"**Counterfactual {cf_idx + 1}:**")
                                    for change in changes:
                                        st.write(change)
                                    st.divider()
                        else:
                            st.warning("No counterfactuals generated")
                except Exception as e:
                    st.error(f"Counterfactual error: {str(e)}")

        # Agreement analysis
        if 'anchor_features' in locals() and anchor_features:
            st.markdown("---")
            st.subheader("📊 Agreement Analysis")
            
            if 'cf_df' in locals() and cf_df is not None:
                disagreements = []
                for col in anchor_features:
                    for cf_idx in range(len(cf_df)):
                        if abs(instance_df.iloc[0][col] - cf_df.iloc[cf_idx][col]) > 1e-6:
                            disagreements.append(col)
                            break
                
                if disagreements:
                    st.error(f"⚠️ Counterfactuals DISAGREE with anchor on: {', '.join(disagreements)}")
                    st.caption("These features are critical decision boundaries")
                else:
                    st.success("✅ Counterfactuals AGREE - they don't change anchor features")
                    st.caption("Anchor features are stable; other features drive the change")

        # Necessity and Sufficiency Tests
        with st.container(border=True):
            nec_and_suf_ui(exp_dice,
                        instance_df,
                        ("opposite" if len(labels)==2 else int(list((st.session_state["model"].steps[-1][1].classes_ if hasattr(st.session_state["model"], "steps") else st.session_state["model"].classes_)).index(desired_class))), 
                        anchor_features)
