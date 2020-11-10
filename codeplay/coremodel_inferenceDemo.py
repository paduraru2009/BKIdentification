# A demo showing core patches results using the model
from coremodelmain import CoreModelMain

if __name__ == "__main__":
    coreModelMain = CoreModelMain()
    coreModelMain.loadDatasets("./") # from local folder

    coreModelMain.loadExistingModels("./demoModels")
    coreModelMain.inferenceDemo((1,1))
    coreModelMain.inferenceDemo((1,2))
    coreModelMain.inferenceDemo((2,1))

