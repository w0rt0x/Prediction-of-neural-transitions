# Testing inner Functions
# - treated as first class objects
def myFun(*argv): 
    for arg in argv: 
        for j in arg:
            print (j)
    
myFun([[2],[2],[4]]) 