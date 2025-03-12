from mlxtend.data import loadlocal_mnist

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import label_binarize
import torch

X_test, y_test = loadlocal_mnist(images_path="data/t10k-images-idx3-ubyte",labels_path="data/t10k-labels-idx1-ubyte")
X_test = X_test.astype("float")/255.0
images, labels = loadlocal_mnist(images_path="data/train-images-idx3-ubyte",labels_path="data/train-labels-idx1-ubyte")
images = images.astype("float")/255.0
# images = torch.load("cgan-mnist-images.pth").detach().cpu().numpy().reshape(-1,784)
# labels = torch.load("cgan-mnist-labels.pth").detach().cpu().numpy()
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_test = label_binarize(y_test, classes=classes)
labels = label_binarize(labels, classes=classes)
classifier = OneVsRestClassifier(MLPClassifier(random_state=30, alpha=1))
# classifier = OneVsRestClassifier(
#             LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=30))
y_score = classifier.fit(images, labels).predict_proba(X_test)

# X_test /= 255
# print(X_test.dtype)
def compute_fpr_tpr_roc(Y_test, Y_score):
    n_classes = Y_score.shape[1]
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for class_cntr in range(n_classes):
        false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(Y_test[:, class_cntr],
                                                                                       Y_score[:, class_cntr])
        roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    return false_positive_rate, true_positive_rate, roc_auc

print(compute_fpr_tpr_roc(y_test,y_score)[2]["micro"])


