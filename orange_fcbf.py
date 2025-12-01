import Orange
from Orange.data import DiscreteVariable
import Orange.preprocess

def fcbf_feature_selection(data, cats, target):
    """ Saves feature reduced csv using FCBF to 'reduced_fcbf.csv'.

    Args:
        data (csv): The input data table.
        cats (list): List of categorical feature names.
        target (str): The target feature name.
    Returns:
        None
    
    """
    oTable = Orange.data.Table(data)

    # Set categorical features (edit domain)
    domain = oTable.domain
    features = []

    for feat in domain.attributes:
        if feat.name in cats:
            features.append(DiscreteVariable(feat.name))
        else:
            features.append(feat)
    
    # Set target feature (select columns)
    targ = domain[target]
    features = [feat for feat in features if feat.name != target]
    new_domain = Orange.data.Domain(features, targ)

    oTable = oTable.transform(new_domain)

    # Discretize continuous (preproccess)
    discretizer = Orange.preprocess.Discretize()
    oTable = discretizer(oTable)

    # Continuize discrete (preproccess)
    continuizer = Orange.preprocess.Continuize()
    oTable = continuizer(oTable)


    # FCBF (Rank)
    fcbf = Orange.preprocess.score.FCBF()
    scores = fcbf(oTable)
    select_idx = [i for i, score in enumerate(scores) if score > (.001**2)]
    selected_features = [oTable.domain.attributes[i] for i in select_idx]
    print("selected features")
    for feat in selected_features:
        print(f"Selected Feature: {feat.name}, Score: {scores[oTable.domain.index(feat)]}")

    reduced_domain = Orange.data.Domain(selected_features, oTable.domain.class_var)
    reduced_data = oTable.transform(reduced_domain)

    # save csv 
    reduced_data.save("reduced_fcbf.csv")
    print(f"Number of Selected features: {len(selected_features)}")
    print(f"Reduced dataset saved to: 'reduced_fcbf.csv'")

if __name__ == "__main__":
    data = "diabetes.csv" 
    cats = ["diabetes","gender","hypertension","heart_disease","smoking_history"]
    target = "diabetes"  

    fcbf_feature_selection(data, cats, target)