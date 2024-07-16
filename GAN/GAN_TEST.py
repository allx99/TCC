import pip
pip.main(['install', 'keras'])
pip.main(['install', 'tensorflow'])
pip.main(['install', 'pandas'])
pip.main(['install', 'numpy'])
pip.main(['install', 'matplotlib'])
pip.main(['install', 'seaborn'])
pip.main(['install', 'scikit-learn'])
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# Assuming preprocessed dataset is loaded as X_train, y_train, X_test, y_test
train_df: pd.DataFrame = pd.read_csv('train.csv')
test_df: pd.DataFrame = pd.read_csv('test.csv')

# Preprocess data
train_df.describe()
test_df.describe()

# Alley identifica o tipo de rua que dá acesso a casa por um beco, quando tem
# Trata trocando o valor NaN por noAlley
train_df['Alley'] = train_df['Alley'].fillna('noAlley')
test_df['Alley'] = test_df['Alley'].fillna('noAlley')

# MasVnrType identifica o tipo de revestimento de alvenaria
# Trata trocando o valor NaN por None
train_df['MasVnrType'] = train_df['MasVnrType'].fillna('None')
test_df['MasVnrType'] = test_df['MasVnrType'].fillna('None')

# MasVnrArea identifica a área de revestimento de alvenaria
# Trata trocando o valor NaN por 0
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(0)

# BsmtQual identifica a altura do porão
# Trata trocando o valor NaN por noBsmt
train_df['BsmtQual'] = train_df['BsmtQual'].fillna('noBsmt')
test_df['BsmtQual'] = test_df['BsmtQual'].fillna('noBsmt')

# FireplaceQu é indicador da qualidade da lareira, quando não tem, está com o valor NaN
# Trata isso colocando noFireplace no lugar
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('noFireplace')
test_df['FireplaceQu'] = test_df['FireplaceQu'].fillna('noFireplace')

# Fence indica a qualidade da cerca, quando não tem está com valor NaN
# Trata isso colocando noFence no lugar
train_df['Fence'] = train_df['Fence'].fillna('noFence')
test_df['Fence'] = test_df['Fence'].fillna('noFence')

# GarageType indica o tipo de garagem, quando não tem está com valor NaN
# Trata isso colocando noGarage no lugar
train_df['GarageType'] = train_df['GarageType'].fillna('noGarage')
test_df['GarageType'] = test_df['GarageType'].fillna('noGarage')

# PoolQC, PoolArea, MiscFeature and MiscVal have too many NaN values and may not correlate with SalePrice
# Handle it by droping those columns from the data
train_df.drop(columns=['PoolQC', 'PoolArea', 'MiscFeature', 'MiscVal'], inplace=True)
test_df.drop(columns=['PoolQC', 'PoolArea', 'MiscFeature', 'MiscVal'], inplace=True)

# MSSubClass is a categorical feature, so it should be treated as such
# Handle it by turning the numbers into letters (e.g. 20 -> A, 30 -> B, etc) and then one-hot encode it
train_df['MSSubClass'] = train_df['MSSubClass'].apply(lambda x: chr(x))
test_df['MSSubClass'] = test_df['MSSubClass'].apply(lambda x: chr(x))
train_df = pd.get_dummies(train_df, columns=['MSSubClass'])
test_df = pd.get_dummies(test_df, columns=['MSSubClass'])

# Some of the categorical features are ordinal and nominal, so they should be treated differently
# Handle all of the nominal features by one-hot encoding them
# MSZoning, Street, Alley, LotConfig, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, Foundation, Heating, Electrical, GarageType, PavedDrive, SaleType, SaleCondition, LandContour,
train_df = pd.get_dummies(train_df, columns=['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition', 'LandContour'])
test_df = pd.get_dummies(test_df, columns=['MSZoning', 'Street', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition', 'LandContour'])

# Drop the Id column, as it is not relevant for the model
train_df.drop(columns=['Id'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

# And drop the columns that were one-hot encoded, as they are not needed anymore
train_df.drop(columns=['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition', 'LandContour'], inplace=True)
test_df.drop(columns=['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition', 'LandContour'], inplace=True)

# Handle all of the ordinal features by mapping them to numbers
# ExterQual, ExterCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, CentralAir, KitchenQual, Functional, FireplaceQu, GarageFinish, GarageQual, GarageCond, Fence, LotShape, Utilities
ordinal_map = {
'Ex': 5,
'Av': 4,
'Gd': 4,
'Mn': 3,
'TA': 3,
'Fa': 2,
'No': 2,
'Po': 1,
'Na': 1,
'GLQ': 6,
'ALQ': 5,
'BLQ': 4,
'Rec': 3,
'LwQ': 2,
'Unf': 1,
'Sal': 0,
'Sev': 1,
'Maj2': 2,
'Maj1': 3,
'Mod': 4,
'Min2': 5,
'Min1': 6,
'Typ': 7,
'Fin': 3,
'RFn': 2,
'Unf': 1,
'GdPrv': 4,
'MnPrv': 3,
'GdWo': 2,
'MnWw': 1,
'Reg': 4,
'IR1': 3,
'IR2': 2,
'IR3': 1,
'AllPub': 4,
'NoSewr': 3,
'NoSeWa': 2,
'ELO': 1,
'noBsmt': 0,
'noFireplace': 0,
'noFence': 0,
'noGarage': 0,
'noPool': 0,
'noMisc': 0,
'noAlley': 0
}
# Apply the mapping to the ordinal features

train_df['ExterQual'] = train_df['ExterQual'].map(ordinal_map)
test_df['ExterQual'] = test_df['ExterQual'].map(ordinal_map)
train_df['ExterCond'] = train_df['ExterCond'].map(ordinal_map)
test_df['ExterCond'] = test_df['ExterCond'].map(ordinal_map)
train_df['BsmtQual'] = train_df['BsmtQual'].map(ordinal_map)
test_df['BsmtQual'] = test_df['BsmtQual'].map(ordinal_map)
train_df['BsmtCond'] = train_df['BsmtCond'].map(ordinal_map)
test_df['BsmtCond'] = test_df['BsmtCond'].map(ordinal_map)
train_df['BsmtExposure'] = train_df['BsmtExposure'].map(ordinal_map)
test_df['BsmtExposure'] = test_df['BsmtExposure'].map(ordinal_map)
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].map(ordinal_map)
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].map(ordinal_map)
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].map(ordinal_map)
test_df['BsmtFinType2'] = test_df['BsmtFinType2'].map(ordinal_map)
train_df['HeatingQC'] = train_df['HeatingQC'].map(ordinal_map)
test_df['HeatingQC'] = test_df['HeatingQC'].map(ordinal_map)
train_df['CentralAir'] = train_df['CentralAir'].map(ordinal_map)
test_df['CentralAir'] = test_df['CentralAir'].map(ordinal_map)
train_df['KitchenQual'] = train_df['KitchenQual'].map(ordinal_map)
test_df['KitchenQual'] = test_df['KitchenQual'].map(ordinal_map)
train_df['Functional'] = train_df['Functional'].map(ordinal_map)
test_df['Functional'] = test_df['Functional'].map(ordinal_map)
train_df['FireplaceQu'] = train_df['FireplaceQu'].map(ordinal_map)
test_df['FireplaceQu'] = test_df['FireplaceQu'].map(ordinal_map)
train_df['GarageFinish'] = train_df['GarageFinish'].map(ordinal_map)
test_df['GarageFinish'] = test_df['GarageFinish'].map(ordinal_map)
train_df['GarageQual'] = train_df['GarageQual'].map(ordinal_map)
test_df['GarageQual'] = test_df['GarageQual'].map(ordinal_map)
train_df['GarageCond'] = train_df['GarageCond'].map(ordinal_map)
test_df['GarageCond'] = test_df['GarageCond'].map(ordinal_map)
train_df['Fence'] = train_df['Fence'].map(ordinal_map)
test_df['Fence'] = test_df['Fence'].map(ordinal_map)
train_df['LotShape'] = train_df['LotShape'].map(ordinal_map)
test_df['LotShape'] = test_df['LotShape'].map(ordinal_map)
train_df['Utilities'] = train_df['Utilities'].map(ordinal_map)
test_df['Utilities'] = test_df['Utilities'].map(ordinal_map)

# Handle remaining NaN values by filling them with the mean of the column
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Handle CentralAir by mapping it to 0 and 1
train_df['CentralAir'] = train_df['CentralAir'].map({'N': 0, 'Y': 1}).fillna(0)
test_df['CentralAir'] = test_df['CentralAir'].map({'N': 0, 'Y': 1}).fillna(0)

# Generate a heatmap to check for correlations
sns.heatmap(train_df.corr())

# Save the heatmap to a file, making it easier to analyze, make it bigger and more readable
# Make the heatmap bigger and save it to a file with a higher resolution and all the values, colors and labels
plt.figure(figsize=(20, 20))
sns.heatmap(train_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.savefig('heatmap.png')

# Split the data into X and y
X_train = train_df.drop(columns=['SalePrice'])
y_train = train_df['SalePrice']
X_test = test_df

# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data to fit the LSTM model
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# y_train shape is (1460,) so it is raising an error because the xpected shape (None, 1, 180)
y_train = y_train.values.reshape(-1, 1, 1)
# Exception encountered when calling LSTMCell.call(). Dimensions must be equal, but are 1 and 180 for ... with input shapes: [1460,1], [180,200]. 
# The error is raised because the input shape of the LSTM layer is (1, 180) and the input shape of the data is (1460, 1)
# The data should be reshaped to (1460, 1, 180) to match the input shape of the LSTM layer



# GAN Model
epochs = 100

# Generator Model
generator = Sequential()
generator.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
generator.add(Dropout(0.2))
generator.add(LSTM(units=50))
generator.add(Dense(units=1))

# Discriminator Model
discriminator = Sequential()
discriminator.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
discriminator.add(Dropout(0.2))
discriminator.add(LSTM(units=50))
discriminator.add(Dense(units=1, activation='sigmoid'))

# Compile both models
generator.compile(optimizer=Adam(), loss='mean_squared_error')
discriminator.compile(optimizer=Adam(), loss='binary_crossentropy')

# Training GAN
for epoch in range(epochs):
    # Train discriminator
    # Real data



    real_data = y_train
    fake_data = generator.predict(X_train)
    discriminator.train_on_batch(real_data, np.ones((real_data.shape[0], 1)))
    discriminator.train_on_batch(fake_data, np.zeros((fake_data.shape[0], 1)))

    # Train generator
    noise = np.random.normal(0, 1, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    generator.train_on_batch(noise, np.ones((noise.shape[0], 1)))

# Evaluate model
predictions = generator.predict(X_test)