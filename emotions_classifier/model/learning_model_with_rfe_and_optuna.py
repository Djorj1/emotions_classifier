import optuna
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

def preprocessing_data(df):
    """Разделение данных на обучающую и тестовую выборки"""
    X = df.drop(columns='target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def objective(trial, X_train, X_test, y_train, y_test):
    """Целевая функция для Optuna оптимизации"""
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs = -1
    )
    
    use_rfe = trial.suggest_categorical('use_rfe', [True, False])

    rfe_step = trial.suggest_int('rfe_step', 1, 10)
    min_features = trial.suggest_int('min_features', 5, min(50, X_train.shape[1]//2), step=5)
        
    rfe = RFECV(
            estimator=estimator,
            step=rfe_step,
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            scoring='f1_weighted',
            min_features_to_select=min_features,
            n_jobs=-1
        )
        
        # Обучаем RFE и преобразуем данные
    X_train_selected = rfe.fit_transform(X_train, y_train)
    X_test_selected = rfe.transform(X_test)
        
        # Обучаем финальную модель на отобранных признаках
    final_estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
    final_estimator.fit(X_train_selected, y_train)
    pred = final_estimator.predict(X_test_selected)
        
        # Сохраняем информацию о количестве признаков
    trial.set_user_attr('n_features_selected', X_train_selected.shape[1])
    

    return -f1_score(y_test, pred, average='weighted')

def optimize_hyperparameters(df, n_trials=100):
    """Основная функция для оптимизации гиперпараметров"""

    X_train, X_test, y_train, y_test = preprocessing_data(df)
    
    
    study = optuna.create_study(direction="minimize") 
    
    
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test), 
        n_trials=n_trials
    )

    rfe_trials = sum(1 for t in study.trials if t.user_attrs.get('use_rfe', False))
    
    return {
        'best_params': study.best_params,
        'best_score': -study.best_value,  
        'study': study
    }



def train_best_model(df, best_params):
    """Обучение модели с лучшими параметрами"""
    X_train, X_test, y_train, y_test = preprocessing_data(df)
    
    use_rfe = best_params.get('use_rfe', False)

    b_params = {
        'n_estimators':best_params['n_estimators'],
        'max_depth':best_params['max_depth'],
        'min_samples_split': best_params['min_samples_split'],
        'min_samples_leaf':best_params['min_samples_leaf'],
        'random_state': 42,
        'n_jobs': -1
    }
    
    rfe = RFECV(estimator=RandomForestClassifier(**b_params),
            step=best_params.get('rfe_step', 5),
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='f1_weighted',
            min_features_to_select=best_params.get('min_features', 10),
            n_jobs=-1)
    
    X_train_selected = rfe.fit_transform(X_train, y_train)
    X_test_selected = rfe.transform(X_test)    

    model = RandomForestClassifier(**b_params)                            
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)
    
    with open('model_rfe.pkl', 'wb') as f:
        pickle.dump(model, f)

    return {
        'model': model,
        'accuracy': accuracy_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions, average='weighted'),
        'classification_report': classification_report(y_test, predictions)
    }