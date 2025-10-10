import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from anchor import anchor_tabular  
import dice_ml

st.header("BoostedXAI: :blue[Anchors] and :orange[Counterfactuals]", divider=True)

# file upload
with st.container(border=True):
    datafile = st.file_uploader("Upload CSV", type=["csv"])

if datafile:
    # Load data
    data = pd.read_csv(datafile)    
    features = data.keys().to_list()

    # target feature 
    with st.container(border=True):
        target = st.segmented_control("Target Feature:", options=features, selection_mode="single", key="target_feature")
        if target:
            st.markdown(f"✅ Target Feature: **{target}**")
            class_names = [str(x) for x in data[target].unique().tolist()]
            st.markdown(f"Detected Classes: {class_names}")

    # categorical features 
    with st.container(border=True):
        categorical_features = st.segmented_control("Categorical Features", options=features, selection_mode="multi")
        st.markdown(f"📊 Selected Categorical Features: {categorical_features if categorical_features else 'None'}")

    # run training only if target chosen 
    if target:
        feature_cols = [c for c in features if c != target]

        X = data[feature_cols].copy()
        y = data[target].copy()

        # Encode categoricals (dummy variables)
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained with accuracy: {acc:.2%}")

        # anchor setup  
        # Identify categorical feature indices
        categorical_indices = []
        categorical_names = {}
        
        for i, col in enumerate(X_train.columns):
            # Check if this was originally categorical
            original_col = col.split('_')[0]  # Handle dummy variables
            if original_col in (categorical_features or []):
                categorical_indices.append(i)
        
        explainer_anchor = anchor_tabular.AnchorTabularExplainer(
            class_names=class_names,
            feature_names=X_train.columns.tolist(),
            train_data=X_train.values,
            categorical_names=categorical_names,
            categorical_features=categorical_indices
        )

        # instance selector
        sample_options = [
            f"Instance {i} - Pred: {str(y_pred[i])}" for i in range(len(y_pred))
        ]
        
        if sample_options:
            selected = st.selectbox("Pick an instance to explain:", sample_options)
            instance_idx = int(selected.split()[1])
            instance = X_test.iloc[instance_idx:instance_idx+1]
            instance_array = instance.values

            st.write("🔎 **Selected Instance:**")
            st.dataframe(instance.T.rename(columns={instance.index[0]: "Value"}), use_container_width=True)

            col1, col2 = st.columns(2)
            
            # anchor explanation 
            with col1:
                st.subheader("⚓ Anchor Explanation")
                with st.spinner("Generating anchor..."):
                    try:
                        exp_anchor = explainer_anchor.explain_instance(
                            instance_array, 
                            model.predict, 
                            threshold=0.95
                        )
                        
                        anchor_names = exp_anchor.names()
                        
                        if anchor_names:
                            st.markdown("**Anchor Rules:**")
                            for i, name in enumerate(anchor_names, 1):
                                st.write(f"{i}. {name}")
                        else:
                            st.info("No strong anchor found")
                        
                        st.metric("Precision", f"{exp_anchor.precision():.2f}")
                        st.metric("Coverage", f"{exp_anchor.coverage():.2f}")
                        
                        # Extract anchor features
                        anchor_features = []
                        for name in anchor_names:
                            feat = name.split()[0]
                            if feat not in anchor_features:
                                anchor_features.append(feat)
                        
                    except Exception as e:
                        st.error(f"Anchor error: {str(e)}")
                        anchor_features = []

            # counterfactual explanation 
            with col2:
                st.subheader("🔄 Counterfactual Explanation")
                
                try:
                    with st.spinner("Generating counterfactuals..."):
                        # Prepare DiCE input
                        train_df = X_train.copy()
                        train_df[target] = y_train.values

                        data_dice = dice_ml.Data(
                            dataframe=train_df,
                            continuous_features=X_train.columns.tolist(),
                            outcome_name=target
                        )
                        
                        model_dice = dice_ml.Model(model=model, backend="sklearn")
                        exp_dice = dice_ml.Dice(data_dice, model_dice, method='random')

                        # Determine desired class
                        n_classes = len(class_names)
                        if n_classes == 2:
                            desired_class = "opposite"
                        else:
                            current_pred = y_pred[instance_idx]
                            desired_class = int((current_pred + 1) % n_classes)

                        cf = exp_dice.generate_counterfactuals(
                            instance, 
                            total_CFs=3, 
                            desired_class=desired_class
                        )

                        cf_df = cf.cf_examples_list[0].final_cfs_df
                        
                        if cf_df is not None and len(cf_df) > 0:
                            st.dataframe(cf_df, use_container_width=True)
                            
                            # Show changes
                            st.markdown("**Key Changes:**")
                            for cf_idx in range(len(cf_df)):
                                changes = []
                                for col in X_train.columns:
                                    original = instance.iloc[0][col]
                                    cf_val = cf_df.iloc[cf_idx][col]
                                    if abs(original - cf_val) > 1e-6:
                                        in_anchor = col in anchor_features
                                        emoji = "❌" if in_anchor else "✅"
                                        changes.append(f"{emoji} {col}: {original:.2f} → {cf_val:.2f}")
                                
                                if changes:
                                    with st.expander(f"CF {cf_idx + 1}"):
                                        for change in changes:
                                            st.write(change)
                        else:
                            st.warning("No counterfactuals generated")
                            
                except Exception as e:
                    st.error(f"Counterfactual error: {str(e)}")

            # analysis 
            if 'anchor_features' in locals() and anchor_features:
                st.markdown("---")
                st.subheader("📊 Agreement Analysis")
                
                if 'cf_df' in locals() and cf_df is not None:
                    disagreements = []
                    for col in anchor_features:
                        for cf_idx in range(len(cf_df)):
                            if abs(instance.iloc[0][col] - cf_df.iloc[cf_idx][col]) > 1e-6:
                                disagreements.append(col)
                                break
                    
                    if disagreements:
                        st.error(f"⚠️ Counterfactuals DISAGREE with anchor on: {', '.join(disagreements)}")
                        st.caption("These features are critical decision boundaries")
                    else:
                        st.success("✅ Counterfactuals AGREE - they don't change anchor features")
                        st.caption("Anchor features are stable; other features drive the change")

else:
    st.info("👆 Upload a CSV file to begin")