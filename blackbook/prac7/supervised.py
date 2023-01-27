import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import datasets


# dataset = pd.read_csv("iris.csv")

dataset = datasets.load_iris()
# dataset.describe()

X=dataset.data[:,[0,1,2,3]]
y=dataset.target
 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = LogisticRegression(random_state=0, solver='lbfgs',multi_class='auto')
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

probs_y = classifier.predict_proba(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)

import seaborn as sns 
import pandas as pd


ax = plt.axes()
df_cm = cm

sns.heatmap(df_cm,annot=True,annot_kws={"size":30},fmt='d',cmap="Blues",ax = ax)
ax.set_title('Confusion Matrix')
plt.show()








