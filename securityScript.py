# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:32:11 2018

@author: Andre Holder

Description: BMI 500 Lab 14: Securities HW
Demonstrate the relationship between error (the difference between the frequency of each attribute's value and 
that which has been introduced to noise) and epsilon

"""

import os
import pandas as pd # Python Data Analysis Library
import numpy as np
import matplotlib.pyplot as plot

### Get adult_gender_race_dataset from current directory to Python environment 
# Read the data from the dataset file (csv format) using pandas and create a dataframe (a table).
curr_dir = os.getcwd()
path_name= curr_dir + "/" + "adult_age_gender_race_dataset.csv"
df = pd.read_csv(path_name, sep=',')

### Setting the column names
# Change column names so they begin with lowercase letters
df.rename(columns=dict(zip(['Age', 'Gender', 'Race'], ['age', 'gender', 'race'])), inplace=True)

### Build histograms for each feature. Note: Add integers to bin numbers to reflect their corresponding values in table
bin_range_age = df['age'].max()-df['age'].min()+1
frequency_age, bins_age = np.histogram(df['age'], bins=bin_range_age) #, range=(df['age'].min(), df['age'].max())
bins_age = np.round(bins_age) #Rounded
# Plot the variable distribution
hist = df.hist(column='age',grid=True,bins=bin_range_age)
plot.title('Distribution of Original Age Attribute')
plot.xlabel('Age')
plot.ylabel('Frequency')
plot.show()

bin_range_race = np.linspace(1,5,5)
frequency_race, bins_race = np.histogram(df['race'], bins=bin_range_race) #, range=(df['age'].min(), df['age'].max())
bins_race = np.round(bins_race) #Rounded
# Plot the variable distribution
hist = df.hist(column='race',bins=bin_range_race)
#plot.set_xticklabels(bin_names)
plot.title('Distribution of Original Race Attribute')
plot.xlabel('Race')
plot.ylabel('Frequency')
plot.show()

bin_range_gender = df['gender'].max()-df['gender'].min()+1
frequency_gender, bins_gender = np.histogram(df['gender'], bins=bin_range_gender) #, range=(df['age'].min(), df['age'].max())
bins_gender = np.round(bins_gender) #Rounded
# Plot the variable distribution
hist = df.hist(column='gender',bins=bin_range_gender)
plot.title('Distribution of Original Gender Attribute')
plot.xlabel('Gender')
plot.ylabel('Frequency')
plot.show()

#Create the error distribution based on the sensitivity and epsilon of the variable (b=sensitivity/epsilon)
def laplaceNoise(x, b):
    x +=  np.random.laplace(0, b)
    return x

# Vectorize Laplace noise
laplaceVectorized = np.vectorize(laplaceNoise)

# Create a list of the errors corresponding to a range of epsilon values for the variable
error_list_age = []
error_list_race = []
error_list_gender = []
epsilon_list = np.linspace(0.1,1,20) # Different epsilon ranges to plot

# Create table of vectorized noisy values in each bin of each variable based on different epsilons
new_frequency_table_age = [] 
sensitivity_age = df['age'].max()
for epsilon in epsilon_list:
    new_frequency_age = laplaceVectorized(frequency_age, sensitivity_age/epsilon)
    new_frequency_table_age.append(new_frequency_age)
    error_age = np.abs(frequency_age-new_frequency_age)
    error_list_age.append(np.mean(error_age)) #Average error for all bins 

new_frequency_table_race = [] 
sensitivity_race = 1
for epsilon in epsilon_list:
    new_frequency_race = laplaceVectorized(frequency_race, sensitivity_race/epsilon)
    new_frequency_table_race.append(new_frequency_race)
    error_race = np.abs(frequency_race-new_frequency_race)
    error_list_race.append(np.mean(error_race))
    
new_frequency_table_gender = [] 
sensitivity_gender = 1
for epsilon in epsilon_list:
    new_frequency_gender = laplaceVectorized(frequency_gender, sensitivity_gender/epsilon)
    new_frequency_table_gender.append(new_frequency_gender)
    error_gender = np.abs(frequency_gender-new_frequency_gender)
    error_list_gender.append(np.mean(error_gender))

### Create plots for each variable showing behavior of epsilon list with each averaged error for all bins
# Two subplots, the axes array is 1-d
f, axarr = plot.subplots(3, sharex=True)
axarr[0].plot(epsilon_list,error_list_age)
axarr[0].set_title('Behavior of Average Age Error with Epsilon')
axarr[1].plot(epsilon_list,error_list_race)
axarr[1].set_title('Behavior of Average Race Error with Epsilon')
axarr[2].plot(epsilon_list,error_list_gender)
axarr[2].set_title('Behavior of Average Gender Error with Epsilon')
plot.xlabel('Epsilon')
plot.ylabel('Error')
plot.show()