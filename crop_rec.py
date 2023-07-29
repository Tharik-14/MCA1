import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

crop = pd.read_csv("Crop_recommendation.csv")

crop.head(5)

crop.label.value_counts()

crop.describe()

crop.shape
crop.apply(lambda x: len(x.isnull()))
# To check wether there null values or not using the useful function Assert
assert crop.isnull().sum().sum() == 0
# Eliminating all duplicated rows with drop_duplicates
crop.drop_duplicates(inplace=True)

# To check wether there duplicated values or not using the useful function Assert
assert crop.duplicated().sum() == 0
# To check wether there unique values in Dataset
crop.apply(lambda x: len(x.unique()))

# all the values in the label are unique
print(crop['label'].unique())
print(" ", len(crop['label'].unique()))

crop.describe()

crop.info()

# ## Exploratory Data Analysis

# 1.  The Temperatures mostly ranged from 15 to 35, which means that most of the plants in the project need at least a
# high or medium temperature.
# The cotton plant is sensitive to heat and cannot bear less than 15 degrees Celsius as it is clear.
# 2.  The Optimum pH for most plants is between 5.5 and 8 , depending on the type of plant
# 3. The Rainfall rates ranged mostly from 50 mm to 150 mm, and this is considered to be a very large amount of rain,
# but there are some plants in the project that need that, The Rainfall here is considered a winter asset, as it works
# to wash away rust in wheat and remove all germs from the plant. Rain increases the efficiency of representation by
# washing leaves and improving the growth of spikes.
#
# 4. As shown in the figure, most of the plants in the project need large amounts of water, and there are some plants
# that need small amounts of water, and it is important not to give too little or too much water. Giving too little
# water will cause the leaves to droop, and too much water will cause the roots to rot.
#
# 5. As shown in the figure, plants that need high or low temperatures, and this is an important point in determining
# when to plant a plant.
#
#

plt.figure(figsize=(17, 9))
plt.subplot(2, 2, 1)
sns.histplot(crop.temperature, binwidth=3.4, color="red", kde={'alpha': 0.5})
plt.subplot(2, 2, 2)
sns.histplot(crop.ph, binwidth=0.4, color="green", kde={'alpha': 0.5})
plt.show()

plt.figure(figsize=(17, 9))
plt.subplot(2, 2, 1)
sns.histplot(crop.rainfall, binwidth=28, color="red", kde={'alpha': 0.5})
plt.subplot(2, 2, 2)
sns.histplot(crop.humidity, binwidth=5, color="green", kde={'alpha': 0.5})

# Checking the Statistics for all the crops
print("Average Ratio of nitrogen in the soil : {0: .2f}".format(crop['N'].mean()))
print("Average Ratio of Phosphorous in the soil : {0: .2f}".format(crop['P'].mean()))
print("Average Ratio of Potassium in the soil : {0: .2f}".format(crop['K'].mean()))
print("Average temperature in Celsius : {0: .2f}".format(crop['temperature'].mean()))
print("Average Relative Humidity in % is : {0: .2f}".format(crop['humidity'].mean()))
print("Average pH value of the soil : {0: .2f}".format(crop['ph'].mean()))
print("Average Rain fall in mm : {0: .2f}".format(crop['rainfall'].mean()))

# The best suitable temperatures for the growth of the mango tree are when the temperature is 30 - 32 ° C
# and it can withstand high temperatures between 44 - 48 ° C. and
# At temperatures of 18 degrees Celsius or less, they die just as mango trees are damaged

plt.figure(figsize=(20, 12))
Crop_Median_Temperatures = crop.query("temperature > 15  & temperature <= 28 ")
sns.histplot(y=Crop_Median_Temperatures.label, shrink=.7, color="red", alpha=0.5)

# Whenever the temperature is more than 20, every time the humidity level increases,
# it is not a requirement to constantly increase, it can be lowered, but it will not drop below 50%
# The Optimum pH for most plants is between 5.5 and 8 , depending on the type of plant

plt.figure(figsize=(20, 6))
Crop_Temperatures = crop.query("temperature < 25 ")
plt.subplot(1, 2, 1)
sns.histplot(Crop_Temperatures.humidity, binwidth=5.1, color="red", kde={'alpha': 0.5}, )
plt.subplot(1, 2, 2)
sns.histplot(Crop_Temperatures.ph, binwidth=0.3, color="green", kde={'alpha': 0.5})
plt.show()

# As shown in the figure, plants that need high or low temperatures, and this is an important point in determining when
# to plant a plant.
plt.figure(figsize=(18, 10))
sns.boxplot(x="temperature", y="label", data=crop,
            whis=[0, 100], width=.6, orient="h")
plt.show()

# In general, when the temperature rises above 21 pH decrease
# The Optimum pH for most plants is between 5.5 and 8 , depending on the type of plant
sns.jointplot(data=crop, x="ph", y="temperature", height=10, hue="label", space=0.1, s=75)
plt.show()

# pH decreases with increase in temperature.
Crop_high_Temperatures = crop.query("temperature > 34 ")
Crop_low_Temperatures = crop.query("temperature < 19.5 ")
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
sns.histplot(Crop_high_Temperatures.ph, binwidth=.34, color="red", alpha=0.5, )
plt.subplot(2, 2, 2)
sns.histplot(Crop_low_Temperatures.ph, binwidth=.34, color="green", alpha=0.5)
plt.show()

print("Lowest pH value at low temperatures : ", Crop_low_Temperatures.ph.min())
print("Lowest pH value at Highest temperatures : ", Crop_high_Temperatures.ph.min())
print("- -- -- --- --- -- --- ")
print("Highest pH value at low temperatures : ", Crop_low_Temperatures.ph.max())
print("Highest pH value at Highest temperatures : ", Crop_high_Temperatures.ph.max())
print("- -- -- --- --- -- --- ")
print("Average pH value at low temperatures : ", Crop_low_Temperatures.ph.mean())
print("Average pH value at Highest temperatures : ", Crop_high_Temperatures.ph.mean())

### Lets understand which crops can only be Grown in Summer Season, Winter Season and Rainy Season
print("Summer Crops")
print(crop[(crop['temperature'] > 30) & (crop['humidity'] > 50)]['label'].unique())
print("-----------------------------------")
print("Winter Crops")
print(crop[(crop['temperature'] < 20) & (crop['humidity'] < 35)]['label'].unique())
print("-----------------------------------")
print("Rainy Crops")
print(crop[(crop['rainfall'] > 200) & (crop['humidity'] > 30)]['label'].unique())

# ### 2-What is the required ratio of potassium content in the soil for some plants?
# Crops which requires very Low Ratio of Potassium Content in Soil:
# ['orange']
#
# Crops which requires very High Ratio of Potassium Content in Soil: ['grapes' 'apple']
#
# Crops which requires median Ratio of Potassium Content in Soil:['chickpea' 'banana' 'watermelon' 'muskmelon' 'papaya']
#


# As shown in the figure, plants that need high or low temperatures, and this is an important point in determining when
# to plant a plant.
plt.figure(figsize=(21, 10))
sns.boxplot(x="K", y="label", data=crop,
            whis=[0, 100], width=0.6
            , orient="h", hue="label", dodge=False)
plt.show()

print("Crops which requires very Low Ratio of Potassium Content in Soil:", crop[crop['K'] < 15]['label'].unique())
print("Crops which requires very High Ratio of Potassium Content in Soil:", crop[crop['K'] > 175]['label'].unique())
print("Crops which requires median Ratio of Potassium Content in Soil:",
      crop.query("K > 45 & K < 90")['label'].unique())

plt.figure(figsize=(30, 13))
sns.violinplot(x="label", y="N", data=crop, hue="label", dodge=False)
plt.show()

print("Crops which requires very Low Ratio of Nitrogen Content in Soil:", crop[crop['N'] < 10]['label'].unique())
print("Crops which requires very High Ratio of Nitrogen Content in Soil:", crop[crop['N'] > 110]['label'].unique())
print("Crops which requires median Ratio of Nitrogen Content in Soil:", crop.query("N > 40 & N < 80")['label'].unique())

plt.figure(figsize=(30, 13))
sns.violinplot(x="label", y="P", data=crop, hue="label", dodge=False)
plt.show()

print("Crops which requires very Low Ratio of Phosphorus Content in Soil:", crop[crop['P'] < 10]['label'].unique())
print("Crops which requires very High Ratio of Phosphorus Content in Soil:", crop[crop['P'] > 110]['label'].unique())
print("Crops which requires median Ratio of Phosphorus Content in Soil:",
      crop.query("P > 70 & P < 110")['label'].unique())

high_weather = crop.query("temperature > 28 & humidity > 50 ")
cold_weather = crop.query("temperature < 20  & humidity < 40")
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(data=high_weather, x="temperature", y="P", s=70, color='red', alpha=0.7)
plt.subplot(1, 2, 2)
sns.scatterplot(data=cold_weather, x="temperature", y="P", s=70, color="green", alpha=0.7)
plt.show()

# ### 5-Does rainfall affect the soil?
# 1- In general, when precipitation increases, it increases with the degree of humidity
#
# 2- Changes in precipitation affect vegetation which has impacts on soil organic matter cycle and the texture of soil.
# This can influence the runoff rate and formation of surface crusts, which affect erosion and cause deterioration
#


# In general, when precipitation increases, it increases with the degree of humidity
plt.figure(figsize=(20, 8))
sns.scatterplot(data=crop, x="rainfall", y="humidity", hue="label", s=70)
plt.show()

high_rainfall = crop.query("rainfall > 100")
low_rainfall = crop.query("rainfall < 80")
print("Lowest humidtiy value at low rainfall : ", low_rainfall.humidity.min())
print("Lowest humidtiy value at Highest rainfall : ", high_rainfall.humidity.min())
print("- -- -- --- --- -- --- ")
print("Highest humidtiy value at low rainfall : ", low_rainfall.humidity.max())
print("Highest humidtiy value at Highest rainfall : ", high_rainfall.humidity.max())
print("- -- -- --- --- -- --- ")
print("Average humidtiy value at low rainfall : ", low_rainfall.humidity.mean())
print("Average humidtiy value at Highest rainfall : ", high_rainfall.humidity.mean())

# ## Correlation between different features


fig, ax = plt.subplots(1, 1, figsize=(17, 8))
sns.heatmap(crop.corr(), annot=True)
plt.title('Correlation between different features', fontsize=15, c='black')
plt.show()

x = crop.drop(['label'], axis=1)
y = crop['label']

print("The Shape of x:", x.shape)
print("The Shape of y:", y.shape)

# ## Feature Selection
# Feature Selection is a techinque of finding out the features that contribute the most to our model i.e. the best
# predictors.


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)

# ## K-Nearest Neighbors
# k- Nearest Neighbors is one of the most basic algorithms used in supervised machine learning. It classifies new data
# points based on similarity index which is usually a distance metric.
# It uses a majority vote will classifying the new data.


# build the KNN model
kn_classifier = KNeighborsClassifier()
kn_classifier.fit(X_train, y_train)

# predict the results
y_pred = kn_classifier.predict(X_test)

pred_kn = kn_classifier.predict(X_test)

# ## Model Evaluation


# print the scores on training and test set
print('Training set score: {:.4f}'.format(kn_classifier.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(kn_classifier.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Confusion Matrix - score : {:.4f}'.format(kn_classifier.score(X_test, y_test))
plt.title(all_sample_title, size=15)
plt.show()

crop.head(5)

newdata = kn_classifier.predict([[60, 55, 44, 23.004459, 82.320763, 7.840207, 263.964248]])
# newdata
