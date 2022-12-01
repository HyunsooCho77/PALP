from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from utils import return_filenames, train_converter, test_converter
import os
import pickle
import numpy as np
from sklearn.covariance import *
import hydra
from omegaconf import DictConfig
from sklearnex import patch_sklearn, config_context
patch_sklearn()


names = [
    "KNN",
    "Logistic regression",
    "Linear SVM",
]

classifiers = [
    KNeighborsClassifier(3),
    LogisticRegression(C=1e5),
    SVC(kernel="linear"),
]



@hydra.main(config_path=".", config_name="model_config.yaml")
def main(args: DictConfig) -> None:
    task_name = args.task_name
    
    train_file, test_file, _ = return_filenames(args, task_name)
    print(f'{train_file} loading representations:')
    with open(os.path.join(test_file), 'rb') as f:
        rep_label = pickle.load(f)

    with open(os.path.join(train_file), 'rb') as f:
        aug_reps = pickle.load(f)

    for name, clf in zip(names, classifiers):
        score_list = []
        train_dset = train_converter(aug_reps, return_type='np_array')
        test_dset = test_converter(rep_label, return_type='np_array')
        X_train, y_train = train_dset
        X_test, y_test = test_dset

        # iterate over classifiers
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        score_list.append(score)
        print(f'{name} AVG score : {sum(score_list)/len(score_list)*100:.2f}')



if __name__ == "__main__":
    main()








