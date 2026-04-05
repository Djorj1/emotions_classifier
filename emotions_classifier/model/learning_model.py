import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier


def train_model(df):
    X = df.drop(columns = 'target')
    y = df['target']
    # print(X.head())


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )



    clf = RandomForestClassifier(n_estimators = 178, max_depth = 16, min_samples_split = 3, min_samples_leaf = 7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    with open('model_big.pkl', 'wb') as f:
        pickle.dump(clf, f)
    return accuracy, f1, cr