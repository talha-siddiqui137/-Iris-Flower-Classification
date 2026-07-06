# TASK : Iris Flower Classification 

# ● Use measurements of Iris flowers (setosa, versicolor, virginica) as input data. 
# ● Train a machine learning model to classify the species based on these measurements. 
# ● Use libraries like Scikit-learn for easy dataset access and model building. 
# ● Evaluate the model’s accuracy and performance using test data. 
# ● Understand basic classification concepts in machine learning.


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("Iris.csv")

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train,y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print(f"Accuracy is : {accuracy * 100:.2f}%")