%matplotlib inline
%reset -f
#for jupyter notebook

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from matplotlib import pyplot
import numpy as np
from datetime import datetime
import pandas as pd
#import seaborn as sns

dataset="dataset3"
numepoch=300 #undefitted; should be 80
batchsize=180
dropoutratio=0.1
modelloss="mse"
modelmetrics=['mean_squared_error', 'mean_absolute_error'] #['accuracy','mse', 'mae', 'mape', 'cosine']
#activationfx='relu'

np.random.seed(7)

def println():
	print "----------------------------------------------------------------------------------------------"

def plot(metric):
	if metric in history.history.keys():
		pyplot.plot(history.history[metric])
		print metric
		pyplot.show()

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,])
        #print predicted[-1]
        curr_frame = curr_frame[6:]
        curr_frame = np.append(curr_frame, [[predicted[-1][0]]],axis=0)
        curr_frame = np.append(curr_frame, [[predicted[-1][1]]],axis=0)
        curr_frame = np.append(curr_frame, [[predicted[-1][2]]],axis=0)
        curr_frame = np.append(curr_frame, [[predicted[-1][3]]],axis=0)
        curr_frame = np.append(curr_frame, [[predicted[-1][4]]],axis=0)
        curr_frame = np.append(curr_frame, [[predicted[-1][5]]],axis=0)
        
    return predicted

'''
Input format:
-14339,11328,5528,9168,215,3468
-13708,679,-4398,14121,-10792,3901
-8640,19080,-5388,14892,-8724,4247
-14422,8136,-2077,17619,-12928,1437
-9103,13786,-3644,15900,-14179,2013
-9947,12121,-5176,11323,-10549,-722
-6665,16036,-2395,9898,-9784,-1606
'''
datasetstr = np.loadtxt(dataset+".csv", delimiter=",", skiprows=0)
datasetfloat = datasetstr.astype(np.float)
allScaler = MinMaxScaler()

datasetscaled =  allScaler.fit_transform(datasetfloat[:,:])

seq_len=48
sequence_length = seq_len + 1
result = []
for index in range(len(datasetscaled) - sequence_length):
	temp=[]
	first=True
	for a in range(sequence_length):
		for x in range(len(datasetscaled[0])):
			temp.append(datasetscaled[index + a][x])
	result.append(temp)
dataset=np.array(result)
 	
predictionsteps=1
t=-1-predictionsteps*5
testingslice = int(len(dataset)*0.8)
trainX = dataset[:testingslice,:t]
trainY = dataset[:testingslice,t:]
testX = dataset[testingslice:,:t]
testY = dataset[testingslice:,t:]
predictX=dataset[testingslice:,:t]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1)) 
print "Dataset loaded."
%notify -o
#%notify -m "Dataset loaded."

model = Sequential()
model.add(LSTM(
    input_dim=1,
    output_dim=24,
    return_sequences=True))
model.add(Dropout(dropoutratio))
model.add(LSTM(
    24,
    return_sequences=False))
model.add(Dropout(dropoutratio))
model.add(Dense(
        output_dim=6))
model.add(Activation("relu"))

model.compile(loss=modelloss, optimizer="rmsprop", metrics=modelmetrics)
#model.compile(optimizer='adam',loss='mse', metrics=['accuracy','mse', 'mae', 'mape', 'cosine'])
print "Model compiled."

#history=model.fit(trainX, trainY, epochs=numepoch, batch_size=batchsize,verbose=0)
history=model.fit(trainX, trainY, epochs=numepoch, batch_size=batchsize,verbose=2) #validation_data=(testX,testY))
print "Model trained."
%notify -o

println()
plot( "acc")
plot("mean_squared_error")
plot("mean_absolute_error")
plot("val_acc")
plot("val_mean_squared_error")
plot("val_mean_absolute_error")
println()

print "\n"
print "Training metrics"
for i in modelmetrics:
    print("%s: %s" % (i, history.history[i][-1]))
print "\n"

'''
pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(history.history['mean_absolute_error'])
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.plot(history.history['cosine_proximity'])
#https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

->model.metrics_names
['loss',
 'acc',
 'mean_squared_error',
 'mean_absolute_error',
 'mean_absolute_percentage_error',
 'cosine_proximity']
'''

score = model.evaluate(testX, testY, batch_size=batchsize)
print "\n"
print "Model tested."
print "Testing metrics"
for i in range(0,len(modelmetrics)):
	print modelmetrics[i]+": "+str(score[i+1])

predictedax = []
predicteday = []
predictedaz = []
predictedgx = []
predictedgy = []
predictedgz = []
for i in range(len(testX)):
    curr_frame=testX[i]
    predictionResult=model.predict(curr_frame[np.newaxis,:,:])[0,]
    predictedax.append(predictionResult[0])
    predicteday.append(predictionResult[1])
    predictedaz.append(predictionResult[2])
    predictedgx.append(predictionResult[3])
    predictedgy.append(predictionResult[4])
    predictedgz.append(predictionResult[5])

print "\n"
print "Doing one step predictions"
predicteday=np.array(predicteday)
predictedax=np.array(predictedax)
predictedaz=np.array(predictedaz)
predictedgy=np.array(predictedgy)
predictedgx=np.array(predictedgx)
predictedgz=np.array(predictedgz)
prediction=np.column_stack((predictedax,predicteday))
prediction=np.column_stack((prediction,predictedaz))
prediction=np.column_stack((prediction,predictedgx))
prediction=np.column_stack((prediction,predictedgy))
prediction=np.column_stack((prediction,predictedgz))
predictedUnpreprocessed =  allScaler.inverse_transform(prediction)
predictedtestY=allScaler.inverse_transform(testY)
exportcsv= np.column_stack((predictedUnpreprocessed,predictedtestY))
xaxis=np.arange(len(predictedUnpreprocessed))
#xaxis=np.column_stack((xaxis,xaxis%24))
final=np.column_stack((xaxis,exportcsv))
np.savetxt('predictionResult.csv', final, delimiter=',',fmt="%0f")#, header=nodeid+"Date,Hour,Predicted_DL,Predicted_UL,Predicted_connmax,Actual_DL,Actual_UL,Actual_connmax")
print "One step predictions outputted"
%notify -o

datacutoff=125
datastart=40

dfax=pd.DataFrame({'x': final[datastart:datacutoff,0], 'y1': final[datastart:datacutoff,1], 'y2': final[datastart:datacutoff,7] })
pyplot.plot( 'x', 'y2', data=dfax, marker='', color='skyblue', linewidth=2, label='Actual Ax')
pyplot.plot( 'x', 'y1', data=dfax, marker='', color='olive', linewidth=2, label='Predicted Ax')
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
pyplot.legend()
pyplot.title("Ax")
pyplot.savefig("Ax.png")
pyplot.show()

dfay=pd.DataFrame({'x': final[datastart:datacutoff,0], 'y1': final[datastart:datacutoff,2], 'y2': final[datastart:datacutoff,8] })
pyplot.plot( 'x', 'y2', data=dfay, marker='', color='skyblue', linewidth=2, label='Actual Ay')
pyplot.plot( 'x', 'y1', data=dfay, marker='', color='olive', linewidth=2, label='Predicted Ay')
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
pyplot.legend()
pyplot.title("Ay")
pyplot.savefig("Ay.png")
pyplot.show()

dfaz=pd.DataFrame({'x': final[datastart:datacutoff,0], 'y1': final[datastart:datacutoff,3], 'y2': final[datastart:datacutoff,9] })
pyplot.plot( 'x', 'y2', data=dfaz, marker='', color='skyblue', linewidth=2, label='Actual Az')
pyplot.plot( 'x', 'y1', data=dfaz, marker='', color='olive', linewidth=2, label='Predicted Az')
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
pyplot.legend()
pyplot.title("Az")
pyplot.savefig("Az.png")
pyplot.show()

dfgx=pd.DataFrame({'x': final[datastart:datacutoff,0], 'y1': final[datastart:datacutoff,4], 'y2': final[datastart:datacutoff,10] })
pyplot.plot( 'x', 'y2', data=dfgx, marker='', color='skyblue', linewidth=2, label='Actual Gx')
pyplot.plot( 'x', 'y1', data=dfgx, marker='', color='olive', linewidth=2, label='Predicted Gx')
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
pyplot.legend()
pyplot.title("Gx")
pyplot.savefig("Gx.png")
pyplot.show()

dfgy=pd.DataFrame({'x': final[datastart:datacutoff,0], 'y1': final[datastart:datacutoff,5], 'y2': final[datastart:datacutoff,11] })
pyplot.plot( 'x', 'y2', data=dfgy, marker='', color='skyblue', linewidth=2, label='Actual Gy')
pyplot.plot( 'x', 'y1', data=dfgy, marker='', color='olive', linewidth=2, label='Predicted Gy')
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
pyplot.legend()
pyplot.title("Gy")
pyplot.savefig("Gy.png")
pyplot.show()

dfgz=pd.DataFrame({'x': final[datastart:datacutoff,0], 'y1': final[datastart:datacutoff,6], 'y2': final[datastart:datacutoff,12] })
pyplot.plot( 'x', 'y2', data=dfgz, marker='', color='skyblue', linewidth=2, label='Actual Gz')
pyplot.plot( 'x', 'y1', data=dfgz, marker='', color='olive', linewidth=2, label='Predicted Gz')
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
pyplot.legend()
pyplot.title("Gz")
pyplot.savefig("Gz.png")
pyplot.show()

dforig=pd.DataFrame({'x': final[datastart:datacutoff,0], 'y1': final[datastart:datacutoff,7], 'y2': final[datastart:datacutoff,8], 'y3': final[datastart:datacutoff,9], 'y4': final[datastart:datacutoff,10] , 'y5': final[datastart:datacutoff,11], 'y6': final[datastart:datacutoff,12]})
pyplot.plot( 'x', 'y2', data=dforig, marker='', color='skyblue', linewidth=2, label='Ay')
pyplot.plot( 'x', 'y1', data=dforig, marker='', color='olive', linewidth=2, label='Ax')
pyplot.plot( 'x', 'y3', data=dforig, marker='', color='r', linewidth=2, label='Az')
pyplot.plot( 'x', 'y4', data=dforig, marker='', color='c', linewidth=2, label='Gx')
pyplot.plot( 'x', 'y5', data=dforig, marker='', color='m', linewidth=2, label='Gy')
pyplot.plot( 'x', 'y6', data=dforig, marker='', color='y', linewidth=2, label='Gz')
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
pyplot.title("Original output")
pyplot.legend()
pyplot.savefig("origoutput.png")
pyplot.show()

dfpredict=pd.DataFrame({'x': final[datastart:datacutoff,0], 'y1': final[datastart:datacutoff,1], 'y2': final[datastart:datacutoff,2], 'y3': final[datastart:datacutoff,3], 'y4': final[datastart:datacutoff,4] , 'y5': final[datastart:datacutoff,5], 'y6': final[datastart:datacutoff,6]})
pyplot.plot( 'x', 'y2', data=dfpredict, marker='', color='skyblue', linewidth=2, label='Ay')
pyplot.plot( 'x', 'y1', data=dfpredict, marker='', color='olive', linewidth=2, label='Ax')
pyplot.plot( 'x', 'y3', data=dfpredict, marker='', color='r', linewidth=2, label='Az')
pyplot.plot( 'x', 'y4', data=dfpredict, marker='', color='c', linewidth=2, label='Gx')
pyplot.plot( 'x', 'y5', data=dfpredict, marker='', color='m', linewidth=2, label='Gy')
pyplot.plot( 'x', 'y6', data=dfpredict, marker='', color='y', linewidth=2, label='Gz')
#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
pyplot.title("Predicted output")
pyplot.legend()
pyplot.savefig("predictedoutput.png")
pyplot.show()

%notify -m "LSTM complete"