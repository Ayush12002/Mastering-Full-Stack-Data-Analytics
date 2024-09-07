#Q1
import numpy as np

# Method 1: Using np.array() with a list of lists
array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Method 1 - Using np.array():")
print(array1)
# Method 2: Using np.zeros() and then setting values
array2 = np.zeros((3, 3), dtype=int)
array2[0, :] = [1, 2, 3]
array2[1, :] = [4, 5, 6]
array2[2, :] = [7, 8, 9]
print("\nMethod 2 - Using np.zeros() and setting values:")
print(array2)
# Method 3: Using np.full() to initialize and then setting specific values
array3 = np.full((3, 3), 0, dtype=int)
array3[0, :] = [1, 2, 3]
array3[1, :] = [4, 5, 6]
array3[2, :] = [7, 8, 9]
print("\nMethod 3 - Using np.full() and setting specific values:")
print(array3)

#Q2
import numpy as np

# Step 1: Generate a 1D array of 100 evenly spaced numbers between 1 and 10
array_1d = np.linspace(1, 10, 100)

# Step 2: Reshape the 1D array into a 2D array
# Reshape to a shape that has the total number of elements equal to 100
array_2d = array_1d.reshape((10, 10))  # For example, reshape to a 10x10 matrix

print("1D Array:")
print(array_1d)
print("\nReshaped 2D Array:")
print(array_2d)


#Q3
#Differences Between np.array, np.asarray, and np.asanyarray

    #np.array
        #Purpose: Creates a new NumPy array.
        ##Behavior: Copies the data by default. If you provide an existing array, it will copy the data unless specified otherwise (e.g., by setting copy=False).
        #Usage: It is the most flexible way to create a new array from a variety of input types (lists, tuples, etc.).
        #Example:
    #import numpy as np
    #a = np.array([1, 2, 3])  # Creates a new array with data copied

#np.asarray

    #Purpose: Converts input to an array without copying if possible.
    #Behavior: It will not copy the data if the input is already a NumPy array. It only converts data types and shapes if necessary. It is used to ensure the input is an array but avoids unnecessary copying.
    #Usage: When you want to make sure the data is a NumPy array, but you don’t want to copy data unnecessarily if it’s already an array.
    #Example:

    #import numpy as np
   # a = np.array([1, 2, 3])
    #b = np.asarray(a)  # No copy is made, b is a view of a

#np.asanyarray

    #Purpose: Converts the input to an array, but it also preserves subclass information.
    #Behavior: It works similarly to np.asarray but preserves subclass types. This is useful when working with arrays that are subclasses of np.ndarray, like np.ma.MaskedArray.
    #Usage: When you need to convert to an array but want to retain any subclass information.
    #Example:

#import numpy as np
#class MyArray(np.ndarray):
   # pass

#a = np.array([1, 2, 3]).view(MyArray)
#b = np.asanyarray(a)  # b will also be of type MyArray


   #Q4
import numpy as np

# Step 1: Generate a 3x3 array with random floating-point numbers between 5 and 20
array = np.random.uniform(5, 20, (3, 3))

# Step 2: Round each number in the array to 2 decimal places
rounded_array = np.round(array, 2)

# Print the result
print("Generated Array:\n", array)
print("\nRounded Array:\n", rounded_array)



#Q5

import numpy as np

# Create a NumPy array with random integers between 1 and 10 of shape (5, 6)
array = np.random.randint(1, 11, size=(5, 6))

# Print the generated array
print("Generated Array:\n", array)

# a) Extract all even integers from the array
even_integers = array[array % 2 == 0]

# b) Extract all odd integers from the array
odd_integers = array[array % 2 != 0]

# Print the results
print("\nEven Integers:\n", even_integers)
print("\nOdd Integers:\n", odd_integers)


#Q6
import numpy as np

# Create a 3D NumPy array of shape (3, 3, 3) containing random integers between 1 and 10
array = np.random.randint(1, 11, size=(3, 3, 3))

# Print the generated array
print("Generated Array:\n", array)

# a) Find the indices of the maximum values along each depth level (third axis)
# Use np.argmax to get the indices of the maximum values along axis 2 (depth level)
max_indices = np.argmax(array, axis=2)

# Print the indices of maximum values
print("\nIndices of Maximum Values:\n", max_indices)

# b) Perform element-wise multiplication of the array with itself
# For element-wise multiplication, we multiply the array by itself
product_array = np.multiply(array, array)

# Print the result of element-wise multiplication
print("\nElement-wise Multiplication Result:\n", product_array)

#Q7

import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Name': ['John Doe', 'Jane Smith', 'Emily Davis'],
    'Phone': ['(123) 456-7890', '987-654-3210', '555-1234'],
    'Age': [28, 34, 45]
}

df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Step 1: Clean the 'Phone' column by removing non-numeric characters
df['Phone'] = df['Phone'].replace({r'\D': ''}, regex=True)

# Step 2: Convert the cleaned 'Phone' column to numeric (integer type)
df['Phone'] = pd.to_numeric(df['Phone'], errors='coerce')

# Display the transformed DataFrame
print("\nTransformed DataFrame:")
print(df)

# Step 3: Display DataFrame attributes and data types of each column
print("\nDataFrame Attributes and Data Types:")
print(df.dtypes)

#Q8
import pandas as pd

# a) Read the 'data.csv' file using pandas, skipping the first 50 rows.
file_path =r'C:\Users\Ayush Kumar\Downloads\People Data.csv' # Adjust this path if needed
df = pd.read_csv(file_path, skiprows=50)

# b) Only read the columns: 'Last Name', 'Gender', 'Email', 'Phone', and 'Salary' from the file.
# If you are reading the file for the first time and want to select specific columns while reading:
df = pd.read_csv(file_path, skiprows=50, usecols=['Last Name', 'Gender', 'Email', 'Phone', 'Salary'])

# c) Display the first 10 rows of the filtered dataset.
print("First 10 rows of the filtered dataset:")
print(df.head(10))

# d) Extract the 'Salary' column as a Series and display its last 5 values.
salary_series = df['Salary']
print("\nLast 5 values of the 'Salary' column:")
print(salary_series.tail(5))


#Q9
import pandas as pd

# Replace with the correct path to your dataset
file_path =r'C:\Users\Ayush Kumar\Downloads\People Data.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the dataset
print(df.head())

# Filter the DataFrame based on the given conditions
filtered_df = df[
    (df['Last Name'].str.contains('Duke', na=False)) &  # Filter rows where 'Last Name' contains 'Duke'
    (df['Gender'].str.contains('Female', na=False)) &  # Filter rows where 'Gender' contains 'Female'
    (df['Salary'] < 8500)  # Filter rows where 'Salary' is less than 8500
]

# Display the filtered DataFrame
print(filtered_df)


#Q10
import pandas as pd
import numpy as np

# Set the seed for reproducibility
np.random.seed(0)

# Generate a Series of 35 random integers between 1 and 6
random_integers = np.random.randint(1, 7, size=35)

# Convert the Series into a 7x5 DataFrame
df = pd.DataFrame(random_integers.reshape(7, 5), columns=[f'Col{i+1}' for i in range(5)])

# Display the DataFrame
print(df)

#Q11
import pandas as pd
import numpy as np

# Set the seed for reproducibility
np.random.seed(0)

# Create the first Series with random numbers between 10 and 50
series1 = pd.Series(np.random.randint(10, 51, size=50), name='col1')

# Create the second Series with random numbers between 100 and 1000
series2 = pd.Series(np.random.randint(100, 1001, size=50), name='col2')

# Combine the Series into a DataFrame
df = pd.concat([series1, series2], axis=1)

# Rename the columns (though they are already named 'col1' and 'col2')
df.columns = ['col1', 'col2']

# Display the DataFrame
print(df)


#Q12
import pandas as pd

# Load the dataset
df = pd.read_csv('people.csv')

# Drop the specified columns
df = df.drop(columns=['Email', 'Phone', 'Date of birth'])

# Drop rows with any missing values
df = df.dropna()

# Print the final DataFrame
print(df)
#Q13
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
x = np.random.rand(100)
y = np.random.rand(100)

# Create a scatter plot
plt.scatter(x, y, color='red', marker='o', label='Data Points')

# Add a horizontal line at y = 0.5
plt.axhline(y=0.5, color='blue', linestyle='--', label='y = 0.5')

# Add a vertical line at x = 0.5
plt.axvline(x=0.5, color='green', linestyle=':', label='x = 0.5')

# Label the x-axis and y-axis
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Set the title
plt.title('Advanced Scatter Plot of Random Values')

# Display the legend
plt.legend()

# Show the plot
plt.show()


#@14
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate a date range
date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Create a DataFrame with random temperature and humidity data
np.random.seed(0)  # For reproducibility
temperature = np.random.uniform(low=0, high=35, size=len(date_range))  # Random temperatures between 0 and 35
humidity = np.random.uniform(low=30, high=100, size=len(date_range))  # Random humidity between 30% and 100%

# Create the DataFrame
df = pd.DataFrame({
    'Date': date_range,
    'Temperature': temperature,
    'Humidity': humidity
    })

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Create a new figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Temperature on the primary y-axis
ax1.plot(df.index, df['Temperature'], color='tab:red', label='Temperature')
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (°C)', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Create a secondary y-axis to plot Humidity
ax2 = ax1.twinx()
ax2.plot(df.index, df['Humidity'], color='tab:blue', label='Humidity')
ax2.set_ylabel('Humidity (%)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Add titles and legends
plt.title('Daily Temperature and Humidity')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()

#Q15
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Generate 1000 samples from a normal distribution
np.random.seed(0)  # For reproducibility
data = np.random.normal(loc=0, scale=1, size=1000)  # mean=0, std_dev=1

# Step 2: Plot the histogram with 30 bins
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')

# Step 3: Overlay the PDF of the normal distribution
mu, std = norm.fit(data)  # Fit a normal distribution to the data
xmin, xmax = plt.xlim()  # Get the limits of the x-axis
x = np.linspace(xmin, xmax, 100)  # Generate values for the x-axis
p = norm.pdf(x, mu, std)  # Compute the PDF values
plt.plot(x, p, 'k', linewidth=2)  # Plot the PDF

# Step 4: Customize the plot
plt.xlabel('Value')
plt.ylabel('Frequency/Probability')
plt.title('Histogram with PDF Overlay')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()

#Q16
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Generate 1000 samples from a normal distribution
np.random.seed(0)  # For reproducibility
data = np.random.normal(loc=0, scale=1, size=1000)  # mean=0, std_dev=1

# Step 2: Plot the histogram with 30 bins
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')

# Step 3: Overlay the PDF of the normal distribution
mu, std = norm.fit(data)  # Fit a normal distribution to the data
xmin, xmax = plt.xlim()  # Get the limits of the x-axis
x = np.linspace(xmin, xmax, 100)  # Generate values for the x-axis
p = norm.pdf(x, mu, std)  # Compute the PDF values
plt.plot(x, p, 'k', linewidth=2)  # Plot the PDF

# Step 4: Customize the plot
plt.xlabel('Value')
plt.ylabel('Frequency/Probability')
plt.title('Histogram with PDF Overlay')  # Set the title

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()

#Q17
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Generate two random arrays
np.random.seed(0)  # For reproducibility
x = np.random.uniform(-10, 10, 100)  # Random x values between -10 and 10
y = np.random.uniform(-10, 10, 100)  # Random y values between -10 and 10

# Determine the quadrant for each point
def determine_quadrant(x, y):
    if x > 0 and y > 0:
        return 'Quadrant 1'
    elif x < 0 and y > 0:
        return 'Quadrant 2'
    elif x < 0 and y < 0:
        return 'Quadrant 3'
    else:
        return 'Quadrant 4'

# Apply the function to the data
quadrants = np.array([determine_quadrant(xi, yi) for xi, yi in zip(x, y)])

# Create a DataFrame for Seaborn
df = pd.DataFrame({
    'X': x,
    'Y': y,
    'Quadrant': quadrants
})

# Step 2: Create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='X', y='Y', hue='Quadrant', palette='viridis', edgecolor='w', s=100)

# Step 3: Customize the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Quadrant-wise Scatter Plot')  # Set the title

# Add a legend
plt.legend(title='Quadrant')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
#Q18
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Title
import numpy as np

# Output to Jupyter Notebook (use output_file() for saving to an HTML file)
output_notebook()

# Step 1: Generate the data
x = np.linspace(0, 2 * np.pi, 100)  # 100 points from 0 to 2*pi
y = np.sin(x)  # Sine wave function

# Step 2: Create a Bokeh plot
p = figure(title="Sine Wave Function", x_axis_label='X', y_axis_label='sin(X)', 
           plot_width=800, plot_height=400)

# Add line renderer
p.line(x, y, line_width=2, color='blue', legend_label='Sine Wave')

# Add grid lines
p.grid.grid_line_color = 'gray'
p.grid.grid_line_alpha = 0.5

# Show the plot
show(p)

#Q19
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import linear_cmap
import numpy as np
import pandas as pd

# Output to Jupyter Notebook (use output_file() for saving to an HTML file)
output_notebook()

# Step 1: Generate random categorical data
np.random.seed(0)  # For reproducibility
categories = [f'Category {i+1}' for i in range(10)]
values = np.random.randint(10, 100, size=len(categories))

# Create a DataFrame
data = pd.DataFrame({
    'Category': categories,
    'Value': values
})

# Convert to ColumnDataSource
source = ColumnDataSource(data=data)

# Step 2: Create a Bokeh bar chart
p = figure(x_range=data['Category'], plot_height=400, plot_width=800, title="Random Categorical Bar Chart",
           x_axis_label='Category', y_axis_label='Value', toolbar_location=None, tools="")

# Color bars based on their values
color_mapper = linear_cmap(field_name='Value', palette='Viridis256', low=min(values), high=max(values))

# Add bars to the plot
p.vbar(x='Category', top='Value', width=0.9, source=source, color=color_mapper, legend_field='Value')

# Add hover tooltips
hover = HoverTool()
hover.tooltips = [("Category", "@Category"), ("Value", "@Value")]
p.add_tools(hover)

# Customize grid and legend
p.grid.grid_line_color = 'gray'
p.grid.grid_line_alpha = 0.5
p.legend.title = 'Value'
p.legend.location = 'top_left'

# Show the plot
show(p)
#Q20
import plotly.graph_objects as go
import numpy as np

# Step 1: Generate random data
np.random.seed(0)  # For reproducibility
x = np.linspace(0, 10, 100)  # 100 points from 0 to 10
y = np.random.random(100)    # Random values between 0 and 1

# Step 2: Create the line plot
fig = go.Figure()

# Add line trace
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Random Data'))

# Step 3: Customize the plot
fig.update_layout(
    title='Simple Line Plot',
    xaxis_title='X-axis',
    yaxis_title='Y-axis'
)

# Show the plot
fig.show()

#Q21
import plotly.graph_objects as go
import numpy as np

# Step 1: Generate random data
np.random.seed(0)  # For reproducibility
labels = [f'Category {i+1}' for i in range(6)]  # Example category labels
values = np.random.randint(10, 100, size=len(labels))  # Random values for each category

# Step 2: Create the pie chart
fig = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    textinfo='label+percent',  # Show labels and percentages
    hole=0.3,  # Optional: Create a donut chart by specifying a hole
    marker=dict(colors=go.colors.sequential.Plasma[len(labels)])  # Color palette
)])

# Step 3: Customize the plot
fig.update_layout(
    title='Interactive Pie Chart'
)

# Show the plot
fig.show()

