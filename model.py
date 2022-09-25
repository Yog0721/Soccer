
import pandas as pd
import numpy as np

spro = pd.read_excel(r"D:\Project 2022\pro saudi.xlsx")

# Bins
Bins = [-1,2,4,6,8,10]
#name of Range
Range_name = ['0-2','2-4','4-6','6-8','8-10']
spro['Goals_range'] = pd.cut(spro['Totalgoals'],Bins,labels=Range_name)

X = spro.iloc[:,[-4,-3,-2]]
y = spro['Goals_range']
X = round(X,2)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


X = norm_func(X)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)


# Model Bulding 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 25)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred


###############################################################################################

acc = []

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])
    
import matplotlib.pyplot as plt

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")


import pickle

# Saving model to disk
pickle.dump(knn, open('D:\Project 2022\Deployment\model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('D:\Project 2022\Deployment\model.pkl','rb'))
print(model.predict([[2, 9, 6]]))

























