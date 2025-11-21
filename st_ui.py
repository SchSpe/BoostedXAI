import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from anchor import anchor_tabular
import dice_ml
from dice_ml import Dice

def parse_anchors_from_rules(rules, feature_names, training_data):
    """Parse anchor bounds from anchor rules.
    
    Args:
        rules: List of anchor rules
        feature_names: List of feature names
        training_data: DataFrame with training data
        
    Returns:
        Dictionary of anchors with 'min' and 'max' bounds for each feature
    """
    anchors = {}
    for feature in feature_names:
        col_min = float(training_data[feature].min())
        col_max = float(training_data[feature].max())
        
        feature_has_rule = False
        
        for rule in rules:
            if feature in rule:
                feature_has_rule = True
                if '>=' in rule:
                    parts = rule.split('>=')
                    val = float(parts[-1].strip())
                    col_min = max(col_min, val)
                elif '>' in rule:
                    parts = rule.split('>')
                    val = float(parts[-1].strip())
                    col_min = max(col_min, val)
                elif '<=' in rule:
                    parts = rule.split('<=')
                    val = float(parts[-1].strip())
                    col_max = min(col_max, val)
                elif '<' in rule:
                    parts = rule.split('<')
                    val = float(parts[-1].strip())
                    col_max = min(col_max, val)
        
        if feature_has_rule:
            if col_min <= col_max:
                anchors[feature] = {'min': col_min, 'max': col_max}
    
    return anchors

st.set_page_config(
    page_title="XAI: Anchors & Counterfactuals",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'target_name' not in st.session_state:
    st.session_state.target_name = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {}
if 'anchors' not in st.session_state:
    st.session_state.anchors = {}
if 'counterfactuals' not in st.session_state:
    st.session_state.counterfactuals = []
if 'anchor_rules' not in st.session_state:
    st.session_state.anchor_rules = []
if 'anchor_precision' not in st.session_state:
    st.session_state.anchor_precision = None
if 'cf_violations' not in st.session_state:
    st.session_state.cf_violations = {}  # Maps CF index to violation status
if 'training_data' not in st.session_state:
    st.session_state.training_data = None  # Store for analysis
if 'anchor_runs' not in st.session_state:
    st.session_state.anchor_runs = []  # Store multiple anchor generation runs

# Title
st.title("🔍 Explainable AI: Anchors & Counterfactuals")
st.markdown("Upload CSV, train sklearn MLP, explore predictions with anchors and counterfactuals")

# Sidebar for data upload and training
with st.sidebar:
    st.header("Data & Model")
    
    # File upload
    uploaded_file = st.file_uploader("1. Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.session_state.data = df
            st.success(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
            
            # Target selection
            target_col = st.selectbox(
                "2. Select Target Column",
                options=[''] + list(df.columns)
            )
            
            if target_col:
                st.session_state.target_name = target_col
                
                # Train button
                if st.button("3. Train MLP Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            # Prepare data
                            X = df.drop(columns=[target_col])
                            y = df[target_col]
                            
                            # Keep only numeric features
                            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
                            X = X[numeric_features]
                            st.session_state.feature_names = numeric_features
                            
                            # Handle missing values
                            X = X.fillna(X.mean())
                            
                            # Store feature statistics
                            feature_stats = {}
                            feature_types = {}
                            for col in numeric_features:
                                feature_stats[col] = {
                                    'min': float(X[col].min()),
                                    'max': float(X[col].max()),
                                    'mean': float(X[col].mean())
                                }
                                
                                # Auto-detect if categorical (few unique values)
                                n_unique = X[col].nunique()
                                if n_unique <= 10:
                                    feature_types[col] = 'categorical'
                                else:
                                    feature_types[col] = 'continuous'
                            
                            st.session_state.feature_stats = feature_stats
                            st.session_state.feature_types = feature_types
                            
                            # Store training data for analysis
                            st.session_state.training_data = X
                            
                            # Classification-only setup
                            st.session_state.class_names = sorted(y.unique().tolist())
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            model = MLPClassifier(
                                hidden_layer_sizes=(64, 32),
                                max_iter=500,
                                random_state=42,
                                early_stopping=True,
                                validation_fraction=0.1
                            )
                            model.fit(X_train_scaled, y_train)
                            accuracy = model.score(X_test_scaled, y_test)
                            
                            st.session_state.model = model
                            st.session_state.scaler = scaler
                            
                            st.success("✓ Classification model trained!")
                            st.metric("Accuracy", f"{accuracy*100:.1f}%")
                            
                            # Initialize feature values to means
                            for col in numeric_features:
                                st.session_state.feature_values[col] = float(X[col].mean())
                        
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
    
    # Model info
    if st.session_state.model is not None:
        st.divider()
        st.subheader("✓ Model Ready")
        st.write("Type: Classification")
        st.write(f"Features: {len(st.session_state.feature_names)}")
        
        st.divider()
        
        # XAI buttons
        st.subheader("🔍 Generate Explanations")
        
        # Anchor generation with multiple runs
        num_anchor_runs = st.slider("Number of anchor runs", 1, 5, 3, key="num_anchor_runs")
        
        if st.button("Generate Anchors", use_container_width=True):
            with st.spinner(f"Computing anchors ({num_anchor_runs} runs)..."):
                try:
                    X = st.session_state.data[st.session_state.feature_names].fillna(
                        st.session_state.data[st.session_state.feature_names].mean()
                    )
                    
                    def predict_fn(instances):
                        X_scaled = st.session_state.scaler.transform(instances)
                        return st.session_state.model.predict(X_scaled)
                    
                    explainer = anchor_tabular.AnchorTabularExplainer(
                        class_names=st.session_state.class_names,
                        feature_names=st.session_state.feature_names,
                        train_data=X.values,
                        categorical_names={}
                    )
                    
                    instance = np.array([st.session_state.feature_values[f] 
                                       for f in st.session_state.feature_names])
                    
                    # Run anchor generation multiple times
                    anchor_runs = []
                    
                    progress_bar = st.progress(0)
                    for run_idx in range(num_anchor_runs):
                        exp = explainer.explain_instance(
                            instance,
                            predict_fn,
                            threshold=0.95,
                            max_anchor_size=3
                        )
                        
                        # Store this run's results
                        run_result = {
                            'rules': exp.names(),
                            'precision': exp.precision(),
                            'coverage': exp.coverage()
                        }
                        anchor_runs.append(run_result)
                        
                        progress_bar.progress((run_idx + 1) / num_anchor_runs)
                    
                    progress_bar.empty()
                    
                    # Store all runs
                    st.session_state.anchor_runs = anchor_runs
                    
                    # Use the first run's anchors as default
                    if anchor_runs:
                        first_run = anchor_runs[0]
                        st.session_state.anchor_rules = first_run['rules']
                        st.session_state.anchor_precision = first_run['precision']
                        st.session_state.anchor_coverage = first_run['coverage']
                    
                    # Parse anchors from first run as default
                    anchors = {}
                    if anchor_runs:
                        first_run = anchor_runs[0]
                        anchors = parse_anchors_from_rules(first_run['rules'], 
                                                          st.session_state.feature_names, 
                                                          X)
                    
                    st.session_state.anchors = anchors
                    
                    if anchors:
                        st.success(f"✓ Anchors generated with {num_anchor_runs} runs!")
                    else:
                        st.warning("Anchor generated but no specific feature constraints found")
                
                except Exception as e:
                    st.error(f"Anchor generation failed: {str(e)}")
        
        num_cfs = st.number_input("Number of counterfactuals", min_value=1, max_value=10, value=3, key="num_cfs")
        
        if st.button("Generate Counterfactuals", use_container_width=True):
            with st.spinner("Computing counterfactuals..."):
                X = st.session_state.data[st.session_state.feature_names].fillna(
                    st.session_state.data[st.session_state.feature_names].mean()
                )
                y = st.session_state.data[st.session_state.target_name]
                
                df_dice = X.copy()
                df_dice[st.session_state.target_name] = y
                
                d = dice_ml.Data(
                    dataframe=df_dice,
                    continuous_features=st.session_state.feature_names,
                    outcome_name=st.session_state.target_name
                )
                
                # Wrapper for classification model
                class SklearnModelWrapper:
                    def __init__(self, model, scaler):
                        self.model = model
                        self.scaler = scaler
                    
                    def predict(self, X):
                        X_scaled = self.scaler.transform(X)
                        return self.model.predict_proba(X_scaled)
                    
                    def predict_proba(self, X):
                        # DiCE sometimes calls this directly
                        return self.predict(X)
                
                model_wrapper = SklearnModelWrapper(
                    st.session_state.model,
                    st.session_state.scaler
                )
                
                m = dice_ml.Model(
                    model=model_wrapper,
                    backend='sklearn',
                    model_type='classifier'
                )
                exp = Dice(d, m, method='random')
                
                query_df = pd.DataFrame([st.session_state.feature_values])
                
                dice_exp = exp.generate_counterfactuals(
                    query_df,
                    total_CFs=num_cfs,
                    desired_class="opposite"
                )
                
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                
                counterfactuals = []
                for idx in range(len(cf_df)):
                    cf = {}
                    for feature in st.session_state.feature_names:
                        cf[feature] = float(cf_df.iloc[idx][feature])
                    counterfactuals.append(cf)
                
                st.session_state.counterfactuals = counterfactuals
                st.success(f"✓ Generated {len(counterfactuals)} counterfactuals!")

# Main content
if st.session_state.model is not None:
    
    # Calculate CF violations first (before visualization)
    if st.session_state.counterfactuals and st.session_state.anchors:
        cf_violations_map = {}
        for cf_idx, cf in enumerate(st.session_state.counterfactuals):
            feature_violations = {}
            
            for feature in st.session_state.feature_names:
                if feature in st.session_state.anchors:
                    anchor = st.session_state.anchors[feature]
                    cf_val = cf[feature]
                    
                    # Check if CF is inside this anchor
                    if anchor['min'] <= cf_val <= anchor['max']:
                        feature_violations[feature] = True  # Inside this anchor
                    else:
                        feature_violations[feature] = False  # Outside this anchor
            
            # This means the CF is in the complete anchor region but has opposite prediction
            if feature_violations:  # Only check if there are any anchors
                violates = all(feature_violations.values())  # Must be inside ALL anchors
            else:
                violates = False
            
            cf_violations_map[cf_idx] = {'violates': violates, 'features': feature_violations}
        
        st.session_state.cf_violations = cf_violations_map
    
    # Feature type configuration
    if st.session_state.feature_names:
        st.divider()
        st.header("🏷️ Feature Types")
        st.markdown("*Auto-detected. Click to override if needed.*")
        
        # Create columns for feature type toggles
        num_features = len(st.session_state.feature_names)
        cols_per_row = 3
        
        for i in range(0, num_features, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < num_features:
                    feature = st.session_state.feature_names[i + j]
                    current_type = st.session_state.feature_types.get(feature, 'continuous')
                    
                    with cols[j]:
                        is_categorical = st.checkbox(
                            f"**{feature}**\nCategorical?",
                            value=(current_type == 'categorical'),
                            key=f"type_{feature}",
                            help=f"Currently: {current_type.title()}"
                        )
                        
                        if is_categorical:
                            st.session_state.feature_types[feature] = 'categorical'
                        else:
                            st.session_state.feature_types[feature] = 'continuous'
        
        st.caption("📌 Categorical features: Use for discrete values (e.g., 0/1, ratings 1-5). Continuous: Use for measurements (e.g., age, income).")
    
    st.divider()
    
    # Visualization
    st.header("📊 Feature Visualization")
    
    if st.session_state.feature_names:
        # Create subplots for all features
        num_features = len(st.session_state.feature_names)
        
        fig = make_subplots(
            rows=1, 
            cols=num_features,
            subplot_titles=[f"{name}{' 🏷️' if st.session_state.feature_types.get(name, 'continuous') == 'categorical' else ''}" 
                          for name in st.session_state.feature_names],
            horizontal_spacing=0.05
        )
        
        for idx, feature in enumerate(st.session_state.feature_names):
            col_num = idx + 1
            
            stats = st.session_state.feature_stats[feature]
            current_val = st.session_state.feature_values[feature]
            feature_type = st.session_state.feature_types.get(feature, 'continuous')
            
            
            # Use the anchors from session state
            display_anchors = st.session_state.anchors
            
            # Background track (gray)
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[stats['min'], stats['max']],
                    mode='lines',
                    line=dict(color='lightgray', width=10),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=col_num
            )
            
            # Anchor region (green rectangle)
            if feature in display_anchors:
                anchor = display_anchors[feature]

                fig.add_trace(
                    go.Scatter(
                        x=[-0.3, 0.3, 0.3, -0.3, -0.3],
                        y=[anchor['min'], anchor['min'], anchor['max'], anchor['max'], anchor['min']],
                        fill='toself',
                        fillcolor='rgba(0, 255, 0, 0.3)',
                        line=dict(color='green', width=0),
                        name='Anchor' if idx == 0 else None,
                        showlegend=(idx == 0),
                        legendgroup='anchor',
                        hovertemplate=f'Anchor: [{anchor["min"]:.2f}, {anchor["max"]:.2f}]<extra></extra>'
                    ),
                    row=1, col=col_num
                )
                    
                # Anchor boundary lines
                fig.add_trace(
                    go.Scatter(
                        x=[-0.5, 0.5],
                        y=[anchor['min'], anchor['min']],
                        mode='lines',
                        line=dict(color='green', width=4),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=col_num
                )
            
                fig.add_trace(
                    go.Scatter(
                        x=[-0.5, 0.5],
                        y=[anchor['max'], anchor['max']],
                        mode='lines',
                        line=dict(color='green', width=4),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=col_num
                )
            
            # Counterfactual points (colored dots with numbers)
            if st.session_state.counterfactuals:
                colors = ['blue', 'purple', 'orange']
                cf_values = [cf[feature] for cf in st.session_state.counterfactuals]
                
                # Calculate horizontal offsets to prevent overlap
                x_positions = []
                for cf_idx, cf_val in enumerate(cf_values):
                    # Check for overlaps with previously placed CFs
                    overlap_threshold = (stats['max'] - stats['min']) * 0.02  # 2% of range
                    x_offset = 0
                    
                    for prev_idx in range(cf_idx):
                        prev_val = cf_values[prev_idx]
                        # If this CF is too close to a previous one
                        if abs(cf_val - prev_val) < overlap_threshold:
                            # Offset alternating left/right
                            if cf_idx % 2 == 1:
                                x_offset = 0.3
                            else:
                                x_offset = -0.3
                            break
                    
                    x_positions.append(x_offset)
                
                # Draw arrows from current value to CFs 
                change_threshold = (stats['max'] - stats['min']) * 0.05
                
                for cf_idx, cf in enumerate(st.session_state.counterfactuals):
                    cf_val = cf[feature]
                    color = colors[cf_idx % len(colors)]
                    x_pos = x_positions[cf_idx]
                    
                    # Calculate change magnitude
                    change_magnitude = abs(cf_val - current_val)
                    
                    # Only draw arrow if change is significant
                    if change_magnitude > change_threshold:
                        # Arrow thickness based on magnitude (normalized)
                        max_change = stats['max'] - stats['min']
                        normalized_change = change_magnitude / max_change
                        arrow_width = 1 + (normalized_change * 4)  # 1-5 width range
                        
                        # Draw arrow line from current to CF
                        fig.add_trace(
                            go.Scatter(
                                x=[0, x_pos],
                                y=[current_val, cf_val],
                                mode='lines',
                                line=dict(
                                    color=color,
                                    width=arrow_width,
                                    dash='dot'
                                ),
                                showlegend=False,
                                hoverinfo='skip',
                                opacity=0.6
                            ),
                            row=1, col=col_num
                        )
                        
                        # Add arrowhead at CF end
                        # Calculate arrow direction
                        arrow_length = 0.03 * (stats['max'] - stats['min'])
                        if cf_val > current_val:
                            # Arrow pointing up
                            arrow_y1 = cf_val - arrow_length
                            arrow_y2 = cf_val - arrow_length
                        else:
                            # Arrow pointing down
                            arrow_y1 = cf_val + arrow_length
                            arrow_y2 = cf_val + arrow_length
                        
                        # Left side of arrowhead
                        fig.add_trace(
                            go.Scatter(
                                x=[x_pos - 0.15, x_pos],
                                y=[arrow_y1, cf_val],
                                mode='lines',
                                line=dict(color=color, width=arrow_width),
                                showlegend=False,
                                hoverinfo='skip',
                                opacity=0.8
                            ),
                            row=1, col=col_num
                        )
                        
                        # Right side of arrowhead
                        fig.add_trace(
                            go.Scatter(
                                x=[x_pos + 0.15, x_pos],
                                y=[arrow_y2, cf_val],
                                mode='lines',
                                line=dict(color=color, width=arrow_width),
                                showlegend=False,
                                hoverinfo='skip',
                                opacity=0.8
                            ),
                            row=1, col=col_num
                        )
                
                # Draw counterfactuals with offsets (after arrows so they appear on top)
                for cf_idx, cf in enumerate(st.session_state.counterfactuals):
                    cf_val = cf[feature]
                    color = colors[cf_idx % len(colors)]
                    x_pos = x_positions[cf_idx]
                    
                    # Check if this CF violates the anchor for this feature
                    violates_this_feature = False
                    overall_violates = False
                    if (cf_idx in st.session_state.cf_violations and 
                        feature in st.session_state.cf_violations[cf_idx]['features']):
                        violates_this_feature = st.session_state.cf_violations[cf_idx]['features'][feature]
                        overall_violates = st.session_state.cf_violations[cf_idx]['violates']
                    
                    # Different styling based on violation status
                    if overall_violates and violates_this_feature:
                        # Red X marker for features that are part of a complete violation
                        marker_symbol = 'x'
                        marker_color = 'red'
                        marker_size = 25
                        border_width = 4
                        border_color = 'darkred'
                        hover_extra = "<br>⚠️ VIOLATES (inside ALL anchor regions)"
                    else:
                        # Normal colored circle (CF respects anchor - outside at least one)
                        marker_symbol = 'circle'
                        marker_color = color
                        marker_size = 20
                        border_width = 2
                        border_color = 'white'
                        hover_extra = ""
                    
                    # Calculate change for hover
                    change_magnitude = abs(cf_val - current_val)
                    change_pct = (change_magnitude / (stats['max'] - stats['min'])) * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[x_pos],
                            y=[cf_val],
                            mode='markers+text',
                            marker=dict(
                                size=marker_size, 
                                color=marker_color, 
                                symbol=marker_symbol,
                                line=dict(color=border_color, width=border_width)
                            ),
                            text=[str(cf_idx + 1)] if not overall_violates else [''],
                            textposition='middle center',
                            textfont=dict(color='white', size=10, family='Arial Black'),
                            name=f'CF {cf_idx + 1}' if idx == 0 else None,
                            showlegend=(idx == 0),
                            legendgroup=f'cf{cf_idx}',
                            hovertemplate=f'CF {cf_idx + 1}: {cf_val:.2f}<br>Change: {change_pct:.1f}%{hover_extra}<extra></extra>'
                        ),
                        row=1, col=col_num
                    )
            
            # Current value (red horizontal line)
            fig.add_trace(
                go.Scatter(
                    x=[-0.5, 0.5],
                    y=[current_val, current_val],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Current' if idx == 0 else None,
                    showlegend=(idx == 0),
                    legendgroup='current',
                    hovertemplate=f'Current: {current_val:.2f}<extra></extra>'
                ),
                row=1, col=col_num
            )
            
            # Set axis properties
            fig.update_xaxes(
                range=[-1, 1],
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=1, col=col_num
            )
            fig.update_yaxes(
                range=[stats['min'] - (stats['max'] - stats['min']) * 0.05, 
                       stats['max'] + (stats['max'] - stats['min']) * 0.05],
                showgrid=True,
                gridcolor='lightgray',
                row=1, col=col_num
            )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add violation summary below visualization
        if st.session_state.cf_violations:
            st.markdown("---")
            violation_summary = []
            respect_summary = []
            
            for cf_idx in range(len(st.session_state.counterfactuals)):
                if cf_idx in st.session_state.cf_violations:
                    if st.session_state.cf_violations[cf_idx]['violates']:
                        violation_summary.append(f"**CF {cf_idx + 1}**")
                    else:
                        respect_summary.append(f"**CF {cf_idx + 1}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if violation_summary:
                    st.error(f"❌ **Violates Anchor:** {', '.join(violation_summary)}")
                    st.caption("Inside ALL anchor constraints with opposite prediction")
                else:
                    st.success("❌ **Violates Anchor:** None")
            
            with col2:
                if respect_summary:
                    st.success(f"✅ **Respects Anchor:** {', '.join(respect_summary)}")
                    st.caption("Outside at least one anchor constraint")
                else:
                    st.info("✅ **Respects Anchor:** None")
    
    st.divider()
    
    # Feature sliders and prediction
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("⚙️ Feature Values")
        
        # Create sliders for each feature
        slider_cols = st.columns(min(5, len(st.session_state.feature_names)))
        
        for idx, feature in enumerate(st.session_state.feature_names):
            col_idx = idx % len(slider_cols)
            with slider_cols[col_idx]:
                stats = st.session_state.feature_stats[feature]
                feature_type = st.session_state.feature_types.get(feature, 'continuous')
                
                # Different slider based on type
                if feature_type == 'categorical':
                    # Discrete slider with integer steps
                    value = st.select_slider(
                        f"{feature} 🏷️",
                        options=sorted(set([int(stats['min']), int(stats['max'])] + 
                                         [int(v) for v in range(int(stats['min']), int(stats['max'])+1)])),
                        value=int(st.session_state.feature_values.get(feature, stats['mean'])),
                        key=f"slider_{feature}"
                    )
                else:
                    # Continuous slider
                    value = st.slider(
                        feature,
                        min_value=stats['min'],
                        max_value=stats['max'],
                        value=st.session_state.feature_values.get(feature, stats['mean']),
                        key=f"slider_{feature}"
                    )
                
                st.session_state.feature_values[feature] = value
    
    with col2:
        st.header("🎯 Prediction")
        
        # Make prediction
        feature_vector = [st.session_state.feature_values[f] for f in st.session_state.feature_names]
        X_pred = np.array([feature_vector])
        X_scaled = st.session_state.scaler.transform(X_pred)
        
        prediction = st.session_state.model.predict(X_scaled)[0]
        probabilities = st.session_state.model.predict_proba(X_scaled)[0]
        predicted_idx = st.session_state.model.classes_.tolist().index(prediction)
        probability = probabilities[predicted_idx]
        
        st.metric("Class", str(prediction))
        st.metric("Confidence", f"{probability*100:.1f}%")
        
        # Show all class probabilities
        with st.expander("All Class Probabilities"):
            for cls, prob in zip(st.session_state.model.classes_, probabilities):
                st.write(f"{cls}: {prob*100:.1f}%")
        
        # Show explanation status
        st.divider()
        st.subheader("Explanations")
        if st.session_state.anchors:
            st.success(f"✓ Anchors: {len(st.session_state.anchors)} features")
        if st.session_state.counterfactuals:
            st.success(f"✓ CFs: {len(st.session_state.counterfactuals)} examples")
    
    # Anchor Rules Display
    if st.session_state.anchor_rules:
        st.divider()
        st.header("🎯 Anchor Rules")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision", f"{st.session_state.anchor_precision:.2%}")
        with col2:
            st.metric("Coverage", f"{st.session_state.anchor_coverage:.2%}")
        
        st.markdown("**Selected Anchor Rules (IF ... THEN prediction holds):**")
        anchor_text = " **AND** ".join([f"`{rule}`" for rule in st.session_state.anchor_rules])
        st.markdown(anchor_text)
        
        # Show individual run results
        if st.session_state.anchor_runs:
            st.markdown("---")
            st.subheader("📊 Anchor Selection")
            
            # Create options with actual anchor rules
            anchor_options = []
            for i, run in enumerate(st.session_state.anchor_runs):
                if run['rules']:
                    # Create a concise string of the rules
                    rules_text = " AND ".join(run['rules'])
                    # Truncate if too long
                    if len(rules_text) > 100:
                        rules_text = rules_text[:97] + "..."
                    option_text = f"{rules_text} (P:{run['precision']:.1%}, C:{run['coverage']:.1%})"
                else:
                    option_text = f"No rules (P:{run['precision']:.1%}, C:{run['coverage']:.1%})"
                anchor_options.append(option_text)
            
            # Radio buttons for anchor selection
            selected_option = st.radio(
                "Choose which anchor rules to display in visualization:",
                options=anchor_options,
                index=0
            )
            
            # Find which run was selected
            run_index = anchor_options.index(selected_option)
            selected_run = st.session_state.anchor_runs[run_index]
            
            # Parse anchors from selected run
            anchors = parse_anchors_from_rules(selected_run['rules'], 
                                              st.session_state.feature_names,
                                              st.session_state.training_data)
            
            # Update session state with selected anchors and rules
            st.session_state.anchors = anchors
            st.session_state.anchor_rules = selected_run['rules']
            st.session_state.anchor_precision = selected_run['precision']
            st.session_state.anchor_coverage = selected_run['coverage']
        
        with st.expander("ℹ️ What do these mean?"):
            st.markdown("""
            - **Precision**: How often the prediction stays the same when these rules are satisfied
            - **Coverage**: What percentage of the dataset satisfies these rules
            - **Rules**: The conditions that define the "anchor" - when these are true, the prediction is stable
            - **Anchor Selection**: Choose which run's anchor rules to display and use for violation checking
            """)

else:
    st.info("👈 Upload a CSV file and train a model to get started")
    
    # Show example
    with st.expander("📖 How to use"):
        st.markdown("""
        ### Steps:
        1. **Upload CSV**: Choose a CSV file with your tabular data
        2. **Select Target**: Pick which column you want to predict
        3. **Train Model**: Click to train an sklearn MLP neural network
        4. **Adjust Features**: Use sliders to change feature values
        5. **Generate Explanations**: 
           - **Anchors**: Show stable regions where prediction doesn't change
           - **Counterfactuals**: Show minimal changes that flip the prediction
        """)
