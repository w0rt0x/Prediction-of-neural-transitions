# Testing inner Functions
# - treated as first class objects
def myFun(*argv): 
    for arg in argv: 
        for j in arg:
            print (j)
    

"""
def plot3D(data, title):
    #Code used from Tutorial:
    #https://pythonprogramming.net/matplotlib-3d-scatterplot-tutorial/

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plt.scatter(data[0][0], data[0][1], data[0][2], color="black", label="Day 1")
    plt.scatter(data[1][0], data[1][1], data[1][2], color="lime", label="Day 2")
    plt.scatter(data[2][0], data[2][1], data[2][2], color="deeppink", label="Day 3")
    plt.scatter(data[3][0], data[3][1], data[3][2], color="darkorange", label="Day 4")

    ax.set_xlabel('Principle Component 1')
    ax.set_ylabel('Principle Component 2')
    ax.set_zlabel('Principle Component 3')
    plt.title(title)
    plt.legend()
    plt.show()
"""

import pandas as pd
  
# initialize list of lists
data = [['tom', 10], ['nick', 15], ['juli', 14]]
  
# Create the pandas DataFrame
df = pd.DataFrame(data, columns = ['Name', 'Age'])
  
# print dataframe.
print(df)