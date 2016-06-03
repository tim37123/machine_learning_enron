from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print X.shape
print X
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print X_new.shape
print X_new