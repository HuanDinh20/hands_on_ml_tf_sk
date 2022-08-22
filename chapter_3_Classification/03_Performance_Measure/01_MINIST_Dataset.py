"""
In this chapter, we will be using the MNIST dataset, which is a set of 70,000 small
images of digits handwritten by high school students and employees of the US Cen‐
sus Bureau. Each image is labeled with the digit it represents. This set has been stud‐
ied so much that it is often called the “Hello World” of Machine Learning: whenever
people come up with a new classification algorithm, they are curious to see how it
will perform on MNIST. Whenever someone learns Machine Learning, sooner or
later they tackle MNIST.
Scikit-Learn provides many helper functions to download popular datasets. MNIST is
one of them.
for the newer version, using load_digits to get this dataset,
and explore new updates
"""
from sklearn.datasets import load_digits
digits = load_digits()

digits_keys = ', '.join(list(digits.keys()))

X, Y = digits['data'], digits['target']
print(""*10)
print("type of digits: \n", type(digits))
print('digits describe: \n', digits['DESCR'])
print("digit keys : \n", digits_keys)
print("data nad target shape: \n")
print((X.shape, Y.shape))


"""
visualize some ditigts
"""
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[2])
plt.show()

