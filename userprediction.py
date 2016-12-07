import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from scipy.sparse import csr_matrix, hstack

 
np.random.seed(123)
datadir = 'data'

#create dataframe from csv data
gender_train = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')
gender_test = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'), index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),  parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), usecols=['event_id','app_id','is_active'], dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))

 
gender_train['trainrow'] = np.arange(gender_train.shape[0])
gender_test['testrow'] = np.arange(gender_test.shape[0])

#create sparse matrix of features

#for phone brand features
brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gender_train['brand'] = phone['brand']
gender_test['brand'] = phone['brand']
trainingMatrix_brand = csr_matrix((np.ones(gender_train.shape[0]),
                       (gender_train.trainrow, gender_train.brand)))
testingMatrix_brand = csr_matrix((np.ones(gender_test.shape[0]),
                       (gender_test.testrow, gender_test.brand)))
print('Phone brand features: training shape {}, testing shape {}'.format(trainingMatrix_brand.shape, testingMatrix_brand.shape))


# for device model features 

modelencoder = LabelEncoder().fit(phone.phone_brand.str.cat(phone.device_model))
phone['model'] = modelencoder.transform(phone.phone_brand.str.cat(phone.device_model))
gender_train['model'] = phone['model']
gender_test['model'] = phone['model']
trainingMatrix_model = csr_matrix((np.ones(gender_train.shape[0]),
                       (gender_train.trainrow, gender_train.model)))
testingMatrix_model = csr_matrix((np.ones(gender_test.shape[0]),
                       (gender_test.testrow, gender_test.model)))
print('Device model features: training shape {}, testing shape {}'.format(trainingMatrix_model.shape, testingMatrix_model.shape))

#for app events data
appevtencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appevtencoder.transform(appevents.app_id)
nbApp = len(appevtencoder.classes_)

#generate device apps by merging events['device_id'] with app_events['device_id']
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gender_train[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gender_test[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
deviceapps.head()

dev = deviceapps.dropna(subset=['trainrow'])
trainingMatrix_app = csr_matrix((np.ones(dev.shape[0]), (dev.trainrow, dev.app)),
                      shape=(gender_train.shape[0],nbApp))
dev = deviceapps.dropna(subset=['testrow'])
testingMatrix_app = csr_matrix((np.ones(dev.shape[0]), (dev.testrow, dev.app)),
                      shape=(gender_test.shape[0],nbApp))
print('Apps features: training shape {}, testing shape {}'.format(trainingMatrix_app.shape, testingMatrix_app.shape))

# for app labels features 
applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appevtencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nbLabel = len(labelencoder.classes_)

#generate device labels by merging app lables with deviceapps df
devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gender_train[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gender_test[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()

dev = devicelabels.dropna(subset=['trainrow'])
trainingMatrix_label = csr_matrix((np.ones(dev.shape[0]), (dev.trainrow, dev.label)),
                      shape=(gender_train.shape[0],nbLabel))
dev = devicelabels.dropna(subset=['testrow'])
testingMatrix_label = csr_matrix((np.ones(dev.shape[0]), (dev.testrow, dev.label)),
                      shape=(gender_test.shape[0],nbLabel))
print('Device labels features: training shape {}, testing shape {}'.format(trainingMatrix_label.shape, testingMatrix_label.shape))


# aggregate features
trainAgg = hstack((trainingMatrix_brand, trainingMatrix_model, trainingMatrix_app, trainingMatrix_label), format='csr')
testAgg =  hstack((testingMatrix_brand, testingMatrix_model, testingMatrix_app, testingMatrix_label), format='csr')
print('Aggregation of all features: training shape {}, testing shape {}'.format(trainAgg.shape, testAgg.shape))


destencoder = LabelEncoder().fit(gender_train.group)
label = destencoder.transform(gender_train.group) 
cat_label = np_utils.to_categorical(label) 


def generateBatch(data, label, size, shuffle):
    nbBatches = np.ceil(data.shape[0]/size)
    counter = 0
    index = np.arange(data.shape[0])
    if shuffle: np.random.shuffle(index)
    while True:
        batch_index = index[size*counter:size*(counter+1)]
        data_batch = data[batch_index,:].toarray()
        label_batch = label[batch_index]
        counter += 1
        yield data_batch, label_batch
        if (counter == nbBatches):
            if shuffle:  np.random.shuffle(index)
            counter = 0

def generateBatchWithoutLabel(data, size, shuffle):
    nbBatches = data.shape[0] / np.ceil(data.shape[0]/size)
    counter = 0
    index = np.arange(data.shape[0])
    while True:
        batch_index = index[size * counter:size * (counter + 1)]
        data_batch = data[batch_index, :].toarray()
        counter += 1
        yield data_batch
        if (counter == nbBatches): counter = 0

# define baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(50, input_dim=trainAgg.shape[1], init='normal', activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(12, init='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])   
    return model

model=baseline_model()

data_train, data_val, label_train, label_val = train_test_split(trainAgg, cat_label, test_size=0.002, random_state=42)

fit= model.fit_generator(generator=generateBatch(data_train, label_train, 32, True),
                         nb_epoch=15,
                         samples_per_epoch=70496,
                         validation_data=(data_val.todense(), label_val), verbose=2
                         )

# evaluate model
scores_val = model.predict_generator(
			generator=generateBatchWithoutLabel(data_val, 32, False), 
			val_samples=data_val.shape[0])
scores = model.predict_generator(
			generator=generateBatchWithoutLabel(testAgg, 32, False), 
			val_samples=testAgg.shape[0])

print('Loss value of evaluation model: {}'.format(log_loss(label_val, scores_val)))
 
predictionDF = pd.DataFrame(scores, index = gender_test.index, columns=destencoder.classes_)
predictionDF
predictionDF.to_csv('Results.csv',index=True)
