from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from Plotter import data_to_dict, get_data, replace_nan, crate_dataframe


def SVM(dictionary, title, kernel, c=1):
    """
    Multi-Class SVM for prediction of Day and Trial
    returns accuracy and confusion matrix
    """

    return 0

def SVM_preprocessing():
    """
    Loads files into dictionary for SVM prediction
    """

    return 0

def prepare_data_PCA():
    return 0

def prepare_data_tSNE():
    return 0

if __name__ == "__main__":
    path = r'C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten'
    dimension = 20
    population = "bl660-1_two_white_Pop01"

    header, data = get_data(r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_class.mat".format(population),
                            r"C:\Users\Sam\Desktop\BachelorInfo\Bachelor-Info\Daten\{}_lact.mat".format(population))
    
    replace_nan(data, 0)
    dictionary = data_to_dict(data, header)

    # Für alle Keys in dict df erstellen und concatinieren
    # https://pythonexamples.org/pandas-concatenate-dataframes/
    df = crate_dataframe(dictionary, stimulus, dimension)
