# The **MNIST Handwritten Digit Classification
#Challenge** is the classic entry point. Image data is generally harder
#to work with than “flat” relational data. The MNIST data is
#beginner-friendly and is small enough to fit on one computer.
#Handwriting recognition will challenge you, but it doesn’t need high
#computational power. Build a neural network from scratch that solves
#the MNIST challenge with high accuracy. **Data Sources** **MNIST
#(http://yann.lecun.com/exdb/mnist/)** – MNIST is a modified subset of
#two datasets collected by the U.S. National Institute of Standards and
#Technology. It contains 70,000 labeled images of handwritten digits.
#**Submission Guidelines** 
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv(r"C:\Users\CHAITANY\Desktop\train.csv").as_matrix()
print(data)

clf=DecisionTreeClassifier()

#training
xtrain = data[0:21000,1:]
train_label=data[0:21000,0]
clf.fit(xtrain,train_label)

#testing_data
xtest=data[21000:,1:]
actual_label=data[21000:,0] #will help in getting the accuracy

print("just for testing")
print("Give any random index from the dataset:")
n=int(input())
d=xtest[n] #randomly checking for the 8th element
d.shape=(28,28)# as x_test was not in a matrix form
print(clf.predict([xtest[n]]))
pt.imshow(255-d,cmap='gray') #black text on white bcakground
pt.show()

p = clf.predict(xtest)

count=0
for i in range(0,21000):
    if p[i]==actual_label[i]:
        count+=1
    else:
        0
         
print("Accuracy:",(count/21000)*100,"%")  

