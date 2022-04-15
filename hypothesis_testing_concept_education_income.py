# -*- coding: utf-8 -*-
"""Hypothesis Testing Concept_education_income.ipynb

Created by:
        Roberd (https://medium.com/@roberdmanihuruk17)

# Does a **college degree** affect people’s *income*?

### Import Libray & Dataset
"""

import pandas as pd
import numpy as np
import scipy.stats as st

from google.colab import drive
drive.mount('/content/gdrive')

"""### Overview Dataset"""

# use "marketing_campaign.csv" dataset

df = pd.read_csv('/content/sample_data/marketing_campaign.csv', sep='\t')
df

df.shape

df.info()

# Check if any value is NaN (empty)

df.isna().sum()[['Education', 'Income']]

# Fill NaN with median
# New data set = df_new

df_new = df.fillna(df['Income'].median())

# Mean old df vs new df after replace NaN with mean

print(np.mean(df['Income']))
print(np.mean(df_new['Income']))

df_new['Education'].value_counts()

"""#### Categorize Education"""

basic = df_new[df_new['Education'] == 'Basic']
graduation = df_new[df_new['Education'] == 'Graduation']
second_cycle = df_new[df_new['Education'] == '2n Cycle']
master = df_new[df_new['Education'] == 'Master']
phd = df_new[df_new['Education'] == 'PhD']

"""### Central Tendency Measurement

#### MEAN
"""

# mean of all data income

df_new['Income'].mean()

# mean of income by each education

df_new.groupby(['Education'])['Income'].mean()

"""#### MEDIAN"""

# mean of all data income

df_new['Income'].median()

# median of income by each education

df_new.groupby(['Education'])['Income'].median()

"""#### MODE"""

# mode of income

st.mode(df_new['Income'])

# mode of income by each education

print(st.mode(basic['Income']))
print(st.mode(graduation['Income']))
print(st.mode(second_cycle['Income']))
print(st.mode(master['Income']))
print(st.mode(phd['Income']))

"""### Spread Measurement

#### VARIANCE
"""

# variance of income

np.var(df_new['Income'])

# variance of income by each education

df_new.groupby(['Education'])['Income'].var()

"""#### STANDARD DEVIATION"""

# standard deviation all income
df_new['Income'].std()

# standard deviation income per category of education
df_new.groupby(['Education'])['Income'].std()

"""#### RANGE"""

# range income
df_new['Income'].max()-df_new['Income'].min()

# range income per education
df_new.groupby(['Education'])['Income'].max()-df_new.groupby(['Education'])['Income'].min()

"""#### QUARTILE"""

# Quartile of income

print("q1 \t\t\t:", df_new['Income'].quantile(0.25))
print("q2 \t\t\t:", df_new['Income'].quantile(0.5))
print("q3 \t\t\t:", df_new['Income'].quantile(0.75))
print("interquartile range \t:", df_new['Income'].quantile(0.75)-df_new['Income'].quantile(0.25))

"""##### Quartile by category education"""

# q1
df_new.groupby(['Education'])['Income'].quantile(0.25)

# q2
df_new.groupby(['Education'])['Income'].quantile(0.5)

# q3
df_new.groupby(['Education'])['Income'].quantile(0.75)

# interquartile
df_new.groupby(['Education'])['Income'].quantile(0.75)-df_new.groupby(['Education'])['Income'].quantile(0.25)

"""### Hypothesis Testing

H0 : college degree **doesn’t affect** people’s income

H1 : college degree **affects** people’s income
"""

basic = df_new[df_new['Education'] == 'Basic']
graduation = df_new[df_new['Education'] == 'Graduation']
second_cycle = df_new[df_new['Education'] == '2n Cycle']
master = df_new[df_new['Education'] == 'Master']
phd = df_new[df_new['Education'] == 'PhD']

anova_test = st.f_oneway(basic['Income'],graduation['Income'],second_cycle['Income'],master['Income'],phd['Income'])

anova_test.pvalue

"""#### Testing Result"""

if anova_test.pvalue > 0.05:
    print('H0 is accepted : college degree doesn’t affect people’s income')
else:
    print('H0 is declined : college degree affect people’s income')
