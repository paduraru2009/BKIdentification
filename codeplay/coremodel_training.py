# A script for doing training for the core model
from coremodelmain import CoreModelMain

if __name__ == "__main__":
	coreModelMain = CoreModelMain()
	coreModelMain.loadDatasets("./")

	# Sample some training data to see if they are correct
	coreModelMain.plotSampleFromDataset(1, 1)
	coreModelMain.plotSampleFromDataset(1, 2)
	coreModelMain.plotSampleFromDataset(2, 1)

	coreModelMain.trainDatasets()
