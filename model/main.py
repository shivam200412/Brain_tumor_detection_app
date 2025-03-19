import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


def get_clean_data():
    data = pd.read_csv(r"F:\VS Code\Brain_tumor_detection_app\data\processed_brain_tumor_data.csv")

    data['Tumor_Type'] = data[['Tumor_Type_Benign', 'Tumor_Type_Malignant']].idxmax(axis=1)
    data['Tumor_Type'] = data['Tumor_Type'].map({'Tumor_Type_Benign': 0, 'Tumor_Type_Malignant': 1})
    data.drop(['Tumor_Type_Benign', 'Tumor_Type_Malignant'], axis=1, inplace=True)

    return data


def create_model(data):
    X = data.drop(['Tumor_Type'], axis=1)
    y = data['Tumor_Type']

    # scale the data
    scaler = StandardScaler()
    X[['Age', 'Tumor_Size']] = scaler.fit_transform(X[['Age', 'Tumor_Size']])

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler



def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    os.makedirs('model', exist_ok=True)
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)



if __name__ == '__main__':
  main()