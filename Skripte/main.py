from Classifier import Classifier
import os
from os.path import isfile, join
import random
from Plotter import Plotter

def get_all_pop(path: str=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten'):
    """
    returns all population-names
    """
    populations = set()
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for i in files:
        if "_class.mat" in i:
            populations.add(i[:-10])

        if "_lact.mat" in i:
            populations.add(i[:-9])
    return list(populations)


#a = Classifier(['bl693_no_white_Pop05', 'bl693_no_white_Pop02', 'bl693_no_white_Pop03'], r'D:\Dataframes\most_active_neurons\40')
#a.split_trial_wise()
#a.use_SMOTE()
#print(a.k_fold_cross_validation_populationwise())
#print(a.k_fold_cross_validation())
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
populations = get_all_pop()
a = Plotter(populations, r'D:\Dataframes\PCA\2')
ok, nt_ok = a.sort_out_populations(show=False)
b = Plotter(ok, r'D:\Dataframes\tSNE\perp30')

b.set_svm_parameter(kernel="linear", c=1.0)
b.CM_for_all_pop("Class-wise Normalized Confusion Matrix of all\n Populations (2 tSNE Components, perplexity=30) with all 4 classes. \n Classification via SVM(kernel='linear', c=1, class_weight='balanced')")
b.set_svm_parameter(kernel="rbf", c=1.0, gamma=0.5)
b.CM_for_all_pop("Class-wise Normalized Confusion Matrix of all\n Populations (2 tSNE Components, perplexity=30) with all 4 classes. \n Classification via SVM(kernel='rbf', c=1, gamma=0.5, class_weight='balanced')")

# PCA
b = Plotter(ok, r'D:\Dataframes\PCA\20')
b.set_svm_parameter("rbf", 1.0, 10, degree=3)
b.CM_for_all_pop("Class-wise Normalized Confusion Matrix of all\n Populations (20 Principle Components) with all 4 classes. \n Classification via SVM(kernel='rbf', c=1, gamma=10, class_weight='balanced')",
                show=False, 
                dest_path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Bachelor-ML\Skripte\Plots\Grid Searches and parameter estimation\SVM\PCA\All-Populations_SVM-rbf-20PCAs.png')

b.set_svm_parameter("poly", 1.0, 1.0, degree=3)
b.CM_for_all_pop("Class-wise Normalized Confusion Matrix of all\n Populations (20 Principle Components) with all 4 classes. \n Classification via SVM(kernel='poly', c=1, degree=3, class_weight='balanced')",
                show=False, 
                dest_path=r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Bachelor-ML\Skripte\Plots\Grid Searches and parameter estimation\SVM\PCA\All-Populations_SVM-poly-20PCAs.png')