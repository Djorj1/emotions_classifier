import optuna
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

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
        random_state=42
    )
    
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    

    return -f1_score(y_test, pred, average='weighted')

def optimize_hyperparameters(df, n_trials=100):
    """Основная функция для оптимизации гиперпараметров"""

    X_train, X_test, y_train, y_test = preprocessing_data(df)
    
    
    study = optuna.create_study(direction="minimize") 
    
    
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test), 
        n_trials=n_trials
    )
    
    
    return {
        'best_params': study.best_params,
        'best_score': -study.best_value,  
        'study': study
    }

def train_best_model(df, best_params):
    """Обучение модели с лучшими параметрами"""
    X_train, X_test, y_train, y_test = preprocessing_data(df)
    
    model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return {
        'model': model,
        'accuracy': accuracy_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions, average='weighted'),
        'classification_report': classification_report(y_test, predictions)
    }