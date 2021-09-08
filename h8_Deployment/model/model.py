import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import make_column_transformer,ColumnTransformer 
import pickle
import numpy as np
df = pd.read_csv('/Users/zaintama/Documents/Hacktiv8/Dataset/Ecommerce.csv')

df = df.drop('ID',inplace=True, axis=1)

X= df.drop('Reached.on.Time_Y.N',axis=1)
y = df['Reached.on.Time_Y.N']


column_trans = make_column_transformer(
    (OneHotEncoder(),['Warehouse_block','Mode_of_Shipment','Product_importance','Gender']),
    remainder='passthrough')

best_pipe_rf = Pipeline([('pre', column_trans),
			('clf', RandomForestClassifier(criterion='entropy', max_depth= 8, min_samples_leaf= 1, min_samples_split= 10))])

#forest_clf = RandomForestClassifier(criterion='entropy',max_depth= 8, min_samples_leaf= 1, min_samples_split= 10)
best_pipe_rf.fit(X, y)

pickle.dump(best_pipe_rf, open('model_classifier', 'wb'))


