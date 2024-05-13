# Penny Shi 
# changge
# pennyshi6678

"""
INSTRUCTIONS

Available: May 2nd

Due: May 12th at 11:59PM

Gentle reminder that, among other things, you

(a) Must answer your questions in the homework3.py file
(b) Must homework3.py commit to your clone of the GitHub homework repo
(c) Must link your GitHub repo to GradeScope
(d) Must NOT repeatedly use a hard-coded path for the working directory
(e) Must NOT modify the original data in any way

Failure to do any of these will result in the loss of points
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


"""
QUESTION 1

In this question, you'll be replicating the graph from Lecture 14, slide 5
which shows the population of Europe from 0 AD to the present day in both
the linear and the log scale. You can find the data in population.csv, and the
variable names are self-explanatory.

Open this data and replicate the graph. 

Clarification: You are not required to replicate the y-axis of the right hand
side graph; leaving it as log values is fine!

Clarification: You are not required to save the figure

Hints: Note that...

- The numpy function .log() can be used to convert a column into logs
- It is a single figure with two subplots, one on the left and the other on
the right
- The graph only covers the period after 0 AD
- The graph only covers Europe
- The figure in the slides is 11 inches by 6 inches
"""
import os
os.getcwd()
os.chdir(r"/Users/changgeshi/Desktop/Harris-Python")

# Join paths:
base_path = (r"/Users/changgeshi/Desktop/Harris-Python")
path = os.path.join(base_path, "population.csv")

# Import csv documents:
df_pop= pd.read_csv(path)

# Filter year after 0 AD column 
df_ad = df_pop[df_pop['Year'] >= 0]

# Select the European countries 
## Import European country code 
df_all = pd.read_csv('all.csv')
condition = df_all['region'] == 'Europe'
filtered_df = df_all.loc[condition, ['region', 'alpha-3']]
filtered_df.rename(columns={'alpha-3': 'Code'}, inplace=True)

## Merge the orignal dataset with the country code 
df_complete = df_ad.merge(filtered_df, on = 'Code')
df_complete.rename(columns = {'Population (historical estimates)': 'pop'}, inplace = True)

## Aggregate the population by year 
df_aggregate = df_complete.groupby('Year')['pop'].sum().reset_index()

## Turn pop into log_pop 
df_aggregate['pop_millions'] = df_aggregate['pop'] / 1_000_000
df_aggregate['log_pop'] = np.log(df_aggregate['pop_millions'])

# Plot the data
# Creating Dual axis
x = df_aggregate['Year']
y = df_aggregate['pop_millions']
a = df_aggregate['log_pop']

# Set the size of the figure
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjusted size for better visualization
ax1, ax2 = axs

# Plot population data on the first axis 
ax1.plot(x, y, "r-", label="population")
ax1.legend(loc="lower right")
ax1.set_xlabel('Year')
ax1.set_title('Population of Europe from 0 AD to the present day in millions')

# Adjust y-axis ticks
def millions_formatter(x, pos):
    return '{:.0f}M'.format(x)

# Plot log of population data on the second axis 
ax2.plot(x, a, "b--", label="population in log")
ax2.legend(loc="lower right")
ax2.set_xlabel('Year')
ax2.set_title('Population of Europe from 0 AD to the present day in millions (log scale)')

# Ajust the layout of the graph
plt.tight_layout()


"""
QUESTION 2

A country's "capital stock" is the value of its' physical capital, which includes the 
stock of equipment, buildings, and other durable goods used in the production 
of goods and services. Macroeconomists seem to conisder it important to have 
public policies that encourage the growth of capital stock. Why is that?

In this exercise we will look at the relationship between capital stock and 
GDP. You can find data from the IMF in "capitalstock.csv" and documentation in
"capitalstock documentation.txt".

In this exercise we will only be using the variables that are demarcated in
thousands of 2017 international dollars to adjust for variation in the value 
of nominal national currency. Hint: These are the the variables that 
end in _rppp.

1. Open the dataset capitalstock.csv and limit the dataframe to only 
observations from 2018

2. Construct a variable called "capital_stock" that is the sum of the general
government capital stock and private capital stock. Drop 
observations where the value of capital stock is 0 or missing. (We will be 
ignoring public-private partnership capital stock for the purpose of t
his exercise.)

3. Create a scatterplot showing the relationship between log GDP and log
capital stock. Put capital stock on the y-axis. Add the line of best 
fit. Add labels where appropriate and make any cosmetic adjustments you want.

(Note: Does this graph suggest that macroeconomists are correct to consider 
 capital stock important? You don't have to answer this question - it's 
 merely for your own edification.)

4. Estimate a model of the relationship between the log of GDP 
and the log of capital stock using OLS. GDP is the dependent 
variable. Print a table showing the details of your model and, using comments, 
interpret the coefficient on capital stock. 

Hint: when using the scatter() method that belongs to axes objects, the alpha
option can be used to make the markers transparent. s is the option that
controls size
"""
# Step 1 
df_capital= pd.read_csv('capitalstock.csv')
df_capital_2018 = df_capital[df_capital['year'] ==2018]

# Step 2
## Calculate the capital stock
df_capital_2018['capital_stock'] = (df_capital_2018['kgov_rppp'] + 
df_capital_2018['kpriv_rppp']) 
## Drpp the N/A and 0 observations from the dataset
df_cleaned = df_capital_2018.dropna().replace(0, pd.NA).dropna()

# Step 3
## Transform GDP and capital stock into log values
df_cleaned['log_GDP'] = np.log(df_cleaned['GDP_rppp'])
df_cleaned['log_capital_stock'] = np.log(df_cleaned['capital_stock'])

## Plotting
x = df_cleaned['log_GDP']
y = df_cleaned['log_capital_stock']
m,b = np.polyfit(x,y,deg = 1)
gen_line = np.poly1d((m,b))

fig, ax = plt.subplots()
ax.scatter(x, y, label='Data')  # Use scatter plot for data points
ax.plot(x, gen_line(x), 'k--', label = 'line of best fit')  # Plot the line of best fit 

# Add legend with label for scatter plot

ax.set_xlabel('GDP in billions (in log)')
ax.set_ylabel('Capital Stock in billions (in Log)')
ax.set_title('The relationship between GDP and Capital Stock')

# Remove borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid
ax.grid(True, linestyle='--', alpha=0.5)  # Add grid with dashed lines and 50% transparency

# Step 4
## Estimate a model of the relationship
model = ols("log_GDP~log_capital_stock", data = df_cleaned)
result = model.fit()
print(result.summary())

## Analysis: for here we can see that the coefficient on log_capital stock is 0.9215, 
## meaning that for every 1% increase in capital stock, the GDP is expected to 
## increase by 0.9215%. There is a strong and positive relationship between capital 
## stock and GDP. The p-value is smaller than 0.05, giving us a statistically significant
## estimation. 


