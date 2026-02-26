import xgboost as xgb

def train_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        scale_pos_weight=(len(y_train)-y_train.sum())/max(y_train.sum(),1)
    )
    model.fit(X_train, y_train)
    return model
