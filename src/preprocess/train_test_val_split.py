from sklearn.model_selection import train_test_split


def train_test_val_split(X, y, test_size=0.2, val_size=0.1, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state)
    return X_train, y_train, X_test, y_test, X_val, y_val
