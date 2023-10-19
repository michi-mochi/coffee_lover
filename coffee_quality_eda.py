#!/usr/bin/env python
# coding: utf-8

# Explore Arabica coffee dataset from [coffee quality database](https://github.com/jldbc/coffee-quality-database/tree/master). This dataset contains reviews of 1312 arabica coffee beans from the Coffee Quality Institute's trained reviewers, and they were collected from the Coffee Quality Institute's review pages in January 2018.

# In[1]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[2]:


import numpy as np
np.bool = np.bool_
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# In[3]:


print(np.__version__)


# In[4]:


df = pd.read_csv("~/Downloads/arabica_data_cleaned.csv") 


# In[5]:


df.shape


# In[6]:


pd.set_option('display.max_columns', None)
df.head()


# In[7]:


df.dtypes


# In[8]:


# check missing values
df.isnull().sum()


# In[9]:


# Extract numerical columns
num_df = df.select_dtypes(include='number')


# In[10]:


num_df.mean().to_frame().T


# In[11]:


num_df.min().to_frame().T


# In[12]:


num_df.max().to_frame().T


# I noticed that some column names containing dots, eg. Country.of.Origin and Number.ofBags
# 
# Row numbers are recorded in 'Unnamed: 0', we could drop this column.
# 
# Also there are quite some missing values in this dataset.
# 
# Average altitude is around 1800, but the maximum value is over 190000. We should take a closer look into altitude columns.

# ## Data Cleaning

# In[13]:


# drop extra row number column
df = df.drop('Unnamed: 0', axis=1)


# In[14]:


# change column names to lower case snake-case
df.columns = (df.columns
                .str.replace('.', '_')
                .str.lower()
             )


# In[15]:


df[df['altitude_low_meters'] >5000]


# According to [Inter American Coffee](https://www.interamericancoffee.com/brazil-cerrado/#:~:text=Brazil%20Cerrado%20coffee%20is%20grown,all%20levels%20of%20the%20production.),  Brazil Cerrado coffee is grown at altitudes of 900 to 1,250 meters above sea level.
# 
# According to [World Data Info](https://www.worlddata.info/america/nicaragua/index.php#:~:text=Nicaragua%20is%20comparatively%20low%20at,islands%20in%20the%20open%20sea.), the highest mountain peak at Nicaragua (Pico Mogotón) is at 2,107 meters.
# 
# According to [Brittanica](https://www.britannica.com/place/Huehuetenango), the highest mountain peak at Huehuetenango is at 1,890 meters.
# 
# According to [Coffee Hero](https://coffeehero.com.au/blogs/news/guatemalan-coffee-beans-everything-you-need-to-know#:~:text=Nuevo%20Oriente%20coffee%20grows%20at,5500%20feet%20above%20sea%20level.&text=The%20Atitlan%20growing%20region%20surrounds,50%20mm%20rain%20every%20month.), Nuevo Oriente coffee grows at high elevations of about 4,300 feet to 5,500 feet above sea level (1,310 to 1,680 meters).
# 
# Looking at altitude column, likely those observations were parsed incorrectly.

# In[16]:


df.at[543, 'altitude_low_meters'] = 1100
df.at[543,'altitude_mean_meters'] = 1100
df.at[543, 'altitude_high_meters'] = 1100

df.at[896, 'altitude_low_meters'] = 1902
df.at[896,'altitude_mean_meters'] = 1902
df.at[896, 'altitude_high_meters'] = 1902

df.at[1040, 'altitude_low_meters'] = 1100
df.at[1040,'altitude_mean_meters'] = 1100
df.at[1040, 'altitude_high_meters'] = 1100

df.at[1144, 'altitude_low_meters'] = 1902
df.at[1144,'altitude_mean_meters'] = 1902
df.at[1144, 'altitude_high_meters'] = 1902


# ### Numerical variables

# In[17]:


plt.hist(df.total_cup_points, bins = 50)
plt.title('Total Cup Points')


# In[18]:


df.total_cup_points.describe()


# The total cup point is normally distributed (bell shaped) with most of the beans are scored between 70 to 90, and an average score of 82
# 

# In[19]:


# Plot rest of numerical variables

fig, axis = plt.subplots(6, 2, figsize=(20,20))
df[['aroma', 'flavor', 'aftertaste', 'acidity','body','balance','uniformity','clean_cup','sweetness','moisture','altitude_low_meters','altitude_high_meters']].hist(ax=axis, edgecolor='black', grid=False, bins = 40)


# Acidity, aroma, body, flavor, aftertaste and balance follow normal distributions.
# 
# Altitudes roughly follow normal distributions, but they have long right tails.
# 
# Clean-cup, sweetness and uniformity are left skewed.
# 
# Moisture follows zero-inflated normal distribution. It's likely that the method to measure moisture was not as refined, which resulted in recording many observations as zero.
# 

# In[20]:


corr = df[['aroma', 'flavor', 'aftertaste', 'acidity','body','balance','uniformity','clean_cup','sweetness','moisture']].corr()
corr


# In[21]:


sns.heatmap(corr, annot=True)
plt.show()


# There are 2 groups of correlated variables:
# 
# (1) Aroma, flavor, aftertaste, acidity, body and balance are strongly correlated (0.7 or above)
# 
# (2) Uniformity, clean cup, sweetness are correlated (~0.5)
# 
# Note that:
# 
# (i) Group 1 and group 2 variables are weakly positively correlated (~0.35)
# 
# (ii) Moisture has weak correlations with group 1 and group 2 variables
# 

# ### Categorical variables

# In[22]:


df.country_of_origin.unique()


# In[23]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(data = df, x = 'country_of_origin').tick_params(axis='x', rotation=90)


# Majority of the coffee beans from the dataset were from Mexico, Guatemala and Colombia.

# In[24]:


sns.boxplot(x="country_of_origin", y="cupper_points", data=df).tick_params(axis='x', rotation=90)


# Coffee beans from Ethiopia and United States have higher cupping points than coffee beans from other origins. From my personal experience, Ethiopia coffee beans usually have stronger and distinct aroma, eg. Yirgacheffe and Worka Sakaro.

# In[25]:


sns.countplot(data = df, x = 'processing_method').tick_params(axis='x', rotation=90)


# Majority of the coffee beans from the dataset were washed/wet processed.

# In[26]:


sns.boxplot(x="processing_method", y="cupper_points", data=df).tick_params(axis='x', rotation=90)


# Washed/wet and Natural/dry proccessed beans have larger variations in cupping scores than other methods because this dataset contains more observations from those two processing methods.

# ### Characteristics, origins and processing methods
# 1. Do coffee beans share similar characteristics(aroma, flavor, aftertaste, acidity, body, balance, uniformity, clean cup and sweetness) if they were processed the same way?
# 
# 2. Do coffee beans from the same origins have similar characteristics?

# #### Processing methods

# ##### Wet process

# In[27]:


corr = df[df['processing_method']=='Washed / Wet'][['aroma', 'flavor', 'aftertaste', 'acidity','body','balance','uniformity','clean_cup','sweetness']].corr()
sns.heatmap(corr, annot=True)
plt.show()


# ##### Dry process

# In[28]:


corr = df[df['processing_method']=='Natural / Dry'][['aroma', 'flavor', 'aftertaste', 'acidity','body','balance','uniformity','clean_cup','sweetness']].corr()
sns.heatmap(corr, annot=True)
plt.show()


# Dry processed beans have stronger correlation between uniformity and clean cup comparing to wet processed beans. 
# 
# Let's zoom into those two variables below:

# In[44]:


sns.kdeplot(df[df['processing_method']=='Washed / Wet']['clean_cup'],label='clean cup')
sns.kdeplot(df[df['processing_method']=='Washed / Wet']['uniformity'], label='uniformity')
plt.title('Washed / Wet Process')
plt.xlabel('score')
plt.legend()
plt.show()


# In[43]:


sns.kdeplot(df[df['processing_method']=='Natural / Dry']['clean_cup'], label='clean cup')
sns.kdeplot(df[df['processing_method']=='Natural / Dry']['uniformity'], label='uniformity')
plt.title('Natural / Dry Process')
plt.xlabel('score')
plt.legend()
plt.show()


# Both processes have similar distributions for clean cup, but washed/wet process has more fluctuations in uniformity while natural/dry process has more stable distribution, which matches the distribution of clean cup.
