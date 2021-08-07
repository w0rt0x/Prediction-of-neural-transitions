from Classifier import Classifier
import os
from os.path import isfile, join
import random


def get_n_random(n, remove=None, path=r'D:\Dataframes\100_Transition'):
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in range(len(files)):
        files[i] = files[i][:-4]
    for i in remove:
        if i in files: files.remove(i)
    test = random.sample(files, n)
    print(test)
    return test

a = Classifier(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], r'D:\Dataframes\most_active_neurons\40')
a.split_trial_wise()
a.use_SMOTE()
#print(a.k_fold_cross_validation_populationwise())
print(a.k_fold_cross_validation())
#a.shuffle_labels()
#a.do_SVM(kernel='rbf',degree=4, c=1, gamma=0.5, class_weight='balanced')
#print("Macro: ",a.get_f1(avg="macro"))
#print("Micro: ", a.get_f1(avg="micro"))
#print("Weighted: ",a.get_f1(avg="weighted"))
#a.plot_CM(title="bl691-2_no_white_Pop01,\n SMOTE on Training-Data\n SVM(kernel='lin',degree=3 c=1, class_weight='balanced')")
#c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 25, 50, 100, 1000]
#c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100]
#title="bl709_one_white_Pop09\n (50 uMAP Components) on SVM (polynomial Kernel,\n class_weight='balanced') and SMOTE on Training-Data"
#a.grid_search(title, C=c, kernel='poly', degree=[2,3,4])