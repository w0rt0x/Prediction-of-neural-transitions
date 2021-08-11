from Classifier import SVMclassifier
from Plotter import Plotter, get_all_pop



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
a = Plotter(populations, r'D:\Dataframes\most_active_neurons\40')
ok, nt_ok = a.sort_out_populations(show=False)

b = Plotter(ok, r'D:\Dataframes\most_active_neurons\40')
b.set_svm_parameter(kernel="rbf", c=1.0, gamma=0.5)
b.boxplot_of_scores("Mean F1-Scores of 5-fold Cross-Validation using the 40 most active neurons (Transitions over 2 Days)\n and a SVM (rbf-Kernel, c=1, gamma=0.5, balanced class weights)")
