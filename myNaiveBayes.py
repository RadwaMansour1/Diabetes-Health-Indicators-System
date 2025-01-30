from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn as sk


class NaiveBayes:
    def __init__(self, df):
        if df is None:
            print("ERORR!")
        else:
            self.df = df
            y = df['Diabetes_binary']  # target
            X = df.drop(['Diabetes_binary'], axis=1)  # data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.25, random_state=0)
            self.model = GaussianNB()
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)

    def getInformation(self):
        print("Accuracy:", sk.metrics.accuracy_score(self.y_test, self.y_pred))
        print("Precision:", sk.metrics.precision_score(self.y_test, self.y_pred))
        print("Recall:", sk.metrics.recall_score(self.y_test, self.y_pred))
        print("F1 score:", sk.metrics.f1_score(self.y_test, self.y_pred))
        cm = sk.metrics.confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        print("\n\n\n\n")
