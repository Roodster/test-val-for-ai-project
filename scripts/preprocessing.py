from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def preprocess(data, target_label='checked'):
    
    # Define your features and target
    X = data.drop(target_label, axis=1)
    y = data[target_label]

    # # Identify numeric and categorical columns
    # numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

    # # Impute and scale numeric features
    # scaler = StandardScaler()
    # for col in numeric_features:
    #     imputer = SimpleImputer(strategy='median')
    #     X[col] = imputer.fit_transform(X[[col]])
    #     X[col] = imputer.transform(X[[col]])
        
    #     X[col] = scaler.fit_transform(X[[col]])
    #     X[col] = scaler.transform(X[[col]])

    return X, y
