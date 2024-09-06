##matplotlip assigment


#Q1
import matplotlib.pyplot as plt

# Step 2: Define data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]

# Step 3: Create scatter plot
plt.scatter(x, y, color='blue', marker='o', label='Data Points')

# Step 4: Customize plot
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot of X vs. Y')
plt.grid(True)
plt.legend()

# Step 5: Display plot
plt.show()
#Q2
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Define data
data = np.array([3, 7, 9, 15, 22, 29, 35])

# Generate x values as the indices of the data array
x_values = np.arange(len(data))

# Step 3: Create line plot
plt.plot(x_values, data, marker='o', color='green', linestyle='-', label='Trend Line')

# Step 4: Customize plot
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of Data Trend')
plt.grid(True)
plt.legend()

# Step 5: Display plot
plt.show()
#Q3
import matplotlib.pyplot as plt

# Step 2: Define data
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 35, 20]

# Step 3: Create bar chart
plt.bar(categories, values, color='skyblue', edgecolor='black')

# Step 4: Customize and Display Plot
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Frequency of Each Category')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Q4
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Generate data
data = np.random.normal(0, 1, 1000)  # Generate 1000 data points from a normal distribution

# Step 3: Bin data
# Calculate the histogram
counts, bin_edges = np.histogram(data, bins=30)

# Step 4: Create bar chart
# Plot bars
plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), edgecolor='black', alpha=0.7)

# Step 5: Customize and Display Plot
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Frequency of Each Value in Data')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#Q5
import matplotlib.pyplot as plt

# Step 2: Define data
sections = ['Section A', 'Section B', 'Section C', 'Section D']
sizes = [25, 30, 15, 30]

# Step 3: Create pie chart
plt.pie(sizes, labels=sections, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])

# Step 4: Customize and Display Plot
plt.title('Percentage Distribution of Sections')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()










##seaborn assigment
#Q1
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Step 2: Generate Synthetic Data
np.random.seed(0)  # For reproducibility
n = 100
x = np.random.normal(loc=5, scale=2, size=n)  # Random values with mean 5 and std deviation 2
y = 2 * x + np.random.normal(loc=0, scale=1, size=n)  # Linear relationship with some noise

# Create a DataFrame for easier handling with Seaborn
data = pd.DataFrame({'X': x, 'Y': y})

# Step 3: Create a Scatter Plot
sns.scatterplot(x='X', y='Y', data=data, color='blue', edgecolor='w', s=100)

# Step 4: Customize and Display the Plot
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Scatter Plot of Synthetic Data')
plt.grid(True)
plt.show()
#Q2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Step 1: Generate a Synthetic Dataset
np.random.seed(0)  # For reproducibility
n = 1000  # Number of data points
data = np.random.normal(loc=50, scale=15, size=n)  # Normal distribution with mean 50 and std deviation 15

# Create a DataFrame for easier handling with Seaborn
df = pd.DataFrame({'Value': data})

# Step 2: Visualize the Distribution
sns.histplot(df['Value'], bins=30, kde=True, color='blue')

# Customize and Display the Plot
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Random Numbers')
plt.grid(True)
plt.show()
#Q3
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Step 1: Create a Dataset
np.random.seed(0)  # For reproducibility

# Define categories and generate random values
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(10, 100, size=(100,))  # Random values between 10 and 100
category_labels = np.random.choice(categories, size=(100,))  # Randomly assign categories

# Create a DataFrame
df = pd.DataFrame({'Category': category_labels, 'Value': values})

# Step 2: Visualize the Comparison

# Bar plot for mean values per category
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Value', data=df, estimator=np.mean, palette='viridis')
plt.xlabel('Category')
plt.ylabel('Mean Value')
plt.title('Mean Value per Category')
plt.grid(True)
plt.show()

# Box plot for distribution of values per category
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Value', data=df, palette='viridis')
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Distribution of Values per Category')
plt.grid(True)
plt.show()

#Q4
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Step 1: Generate a Dataset
np.random.seed(0)  # For reproducibility

# Define categories and generate random values
categories = ['Category A', 'Category B', 'Category C', 'Category D']
n_samples = 200
values = np.random.normal(loc=50, scale=15, size=n_samples)  # Normal distribution
category_labels = np.random.choice(categories, size=n_samples)  # Randomly assign categories

# Create a DataFrame
df = pd.DataFrame({'Category': category_labels, 'Value': values})

# Step 2: Visualize the Distribution

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Value', data=df, palette='coolwarm')
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Distribution of Values Across Categories')
plt.grid(True)
plt.show()

# Violin Plot (Alternative)
plt.figure(figsize=(10, 6))
sns.violinplot(x='Category', y='Value', data=df, palette='coolwarm')
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Distribution of Values Across Categories (Violin Plot)')
plt.grid(True)
plt.show()
#Q5
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Data
np.random.seed(0)  # For reproducibility

# Generate random data
n_samples = 100
n_features = 5

# Create a base feature with random values
base_feature = np.random.normal(0, 1, n_samples)

# Create correlated features by adding noise to the base feature
data = {
    'Feature 1': base_feature,
    'Feature 2': base_feature + np.random.normal(0, 0.5, n_samples),
    'Feature 3': base_feature + np.random.normal(0, 1, n_samples),
    'Feature 4': base_feature + np.random.normal(0, 1.5, n_samples),
    'Feature 5': np.random.normal(0, 1, n_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Step 2: Compute the Correlation Matrix
correlation_matrix = df.corr()

# Step 3: Visualize the Correlation Matrix using a Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()




#PLOTLY ASSIGNMENT
#Q1
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Set seed for reproducibility
np.random.seed(30)

# Generate the data
data = {
    'X': np.random.uniform(-10, 10, 300),
    'Y': np.random.uniform(-10, 10, 300),
    'Z': np.random.uniform(-10, 10, 300)
}
df = pd.DataFrame(data)

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=df['X'],
    y=df['Y'],
    z=df['Z'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['Z'], # Color by Z values
        colorscale='Viridis', # Color scale
        opacity=0.8
    )
)])

# Update layout
fig.update_layout(
    title='3D Scatter Plot',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

# Show the plot
fig.show()
#Q2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(15)

# Generate the data
data = {
    'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200),
    'Score': np.random.randint(50, 100, 200)
}
df = pd.DataFrame(data)

# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Grade', y='Score', data=df, palette='muted')

# Update labels and title
plt.xlabel('Grade')
plt.ylabel('Score')
plt.title('Distribution of Scores Across Different Grades')

# Show the plot
plt.show()
#generate the data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(20)

# Generate the data
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)

# prepare the data for heatmap
# Pivot the DataFrame to get the heatmap format
pivot_table = df.pivot_table(values='Sales', index='Day', columns='Month', aggfunc='mean', fill_value=0)

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='g')

# Update labels and title
plt.xlabel('Month')
plt.ylabel('Day')
plt.title('Sales Data Heatmap Across Months and Days')

# Show the plot
plt.show()
#Q3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(20)

# Generate the data
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)
# Pivot the DataFrame to get the heatmap format
pivot_table = df.pivot_table(values='Sales', index='Day', columns='Month', aggfunc='mean', fill_value=0)

# Create a heatmap
plt.figure(figsize=(12, 8))  # Set the figure size
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='.0f', linewidths=.5)  # Customize the heatmap appearance

# Update labels and title
plt.xlabel('Month')
plt.ylabel('Day')
plt.title('Sales Data Heatmap Across Months and Days')

# Show the plot
plt.show()


#Q4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate x and y data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Compute z based on the function z = sin(sqrt(x^2 + y^2))
z = np.sin(np.sqrt(x**2 + y**2))

# Flatten the arrays and create a DataFrame
data = {
    'X': x.flatten(),
    'Y': y.flatten(),
    'Z': z.flatten()
}
df = pd.DataFrame(data)

# Prepare the plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Extract x, y, and z from the DataFrame
x = df['X'].values.reshape(100, 100)
y = df['Y'].values.reshape(100, 100)
z = df['Z'].values.reshape(100, 100)

# Plot the surface
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Add color bar which maps values to colors
fig.colorbar(surf, aspect=5)

# Set labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface Plot of z = sin(sqrt(x^2 + y^2))')

# Show the plot
plt.show()
#Q5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(25)

# Generate the dataset
data = {
    'Country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
    'Population': np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000, 5)
}
df = pd.DataFrame(data)

# Prepare the bubble chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the bubble chart
bubble = ax.scatter(
    df['GDP'],              # x-axis
    df['Population'],       # y-axis
    s=df['Population']*2,   # Bubble size (scaled for visibility)
    alpha=0.5,              # Transparency of the bubbles
    c=df['Population'],     # Color by population (optional)
    cmap='viridis',         # Color map
    edgecolor='w',          # Bubble border color
    linewidth=0.5          # Bubble border width
)

# Add labels for each country
for i, country in enumerate(df['Country']):
    ax.text(
        df['GDP'][i], 
        df['Population'][i], 
        country, 
        fontsize=9,
        ha='right'
    )

# Set labels and title
ax.set_xlabel('GDP (in billions)')
ax.set_ylabel('Population (in millions)')
ax.set_title('Bubble Chart of Country GDP and Population')

# Add a color bar to represent population
cbar = plt.colorbar(bubble, ax=ax)
cbar.set_label('Population')
cbar.set_ticks([df['Population'].min(), df['Population'].max()])
cbar.set_ticklabels([str(df['Population'].min()), str(df['Population'].max())])

# Show the plot
plt.grid(True)
plt.show()





#BOKEH ASSIGNMENT


#Q1
from bokeh.plotting import figure, show, output_file
import numpy as np

# Generate x-values from 0 to 10
x = np.linspace(0, 10, 100)
# Compute y-values as the sine of x
y = np.sin(x)

# Create a new plot with title and axis labels
p = figure(title="Sine Wave", x_axis_label='x', y_axis_label='y', plot_width=800, plot_height=400)

# Add a line renderer with the sine wave data
p.line(x, y, legend_label="Sine Wave", line_width=2)

# Specify the output file (this will save the plot as an HTML file)
output_file("sine_wave.html")

# Show the plot
show(p)
#Q2

from bokeh.plotting import figure, show, output_file
import numpy as np
import pandas as pd

# Generate random x and y values
np.random.seed(42)  # For reproducibility
x = np.random.uniform(0, 10, 100)
y = np.random.uniform(0, 10, 100)

# Generate random sizes and colors
sizes = np.random.uniform(5, 50, 100)  # Marker sizes between 5 and 50
colors = np.random.rand(100)  # Random colors between 0 and 1

# Create a DataFrame (optional, but good for handling data)
df = pd.DataFrame({'x': x, 'y': y, 'sizes': sizes, 'colors': colors})

# Create a new plot with title and axis labels
p = figure(title="Scatter Plot with Variable Sizes and Colors", x_axis_label='x', y_axis_label='y', plot_width=800, plot_height=400)

# Add a scatter renderer with the data
p.scatter(x='x', y='y', size='sizes', color='colors', source=df, legend_label="Data Points", fill_alpha=0.6)

# Specify the output file (this will save the plot as an HTML file)
output_file("scatter_plot.html")

# Show the plot
show(p)
#Q3
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
import pandas as pd

# Dataset
fruits = ['Apples', 'Oranges', 'Bananas', 'Pears']
counts = [20, 25, 30, 35]

# Create a DataFrame
df = pd.DataFrame({
    'Fruit': fruits,
    'Count': counts
})

# Create a new plot
p = figure(x_range=df['Fruit'], plot_height=350, title="Fruit Counts",
           toolbar_location=None, tools="")

# Add bars to the plot
p.vbar(x='Fruit', top='Count', width=0.9, source=df, legend_field="Fruit",
       line_color='white', fill_color='blue')

# Set axis labels
p.xaxis.axis_label = "Fruit"
p.yaxis.axis_label = "Count"
p.xaxis.major_label_orientation = 1.2  # Rotate x-axis labels for better readability

# Remove the grid lines for the x-axis
p.xgrid.grid_line_color = None

# Add title and legend
p.title.text_font_size = '16pt'
p.legend.title = "Fruits"
p.legend.label_text_font_size = '12pt'
p.legend.location = 'top_left'

# Specify the output file (this will save the plot as an HTML file)
output_file("bar_chart.html")

# Show the plot
show(p)
#Q4
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook

# Generate random data
np.random.seed(42)
data_hist = np.random.randn(1000)

# Compute histogram
hist, edges = np.histogram(data_hist, bins=30)

# Create a new plot
p = figure(title="Histogram of Random Data", x_axis_label='Value', y_axis_label='Frequency',
           plot_height=400, plot_width=600)

# Add bars to the plot
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='blue', line_color='black')

# Specify the output file (this will save the plot as an HTML file)
output_file("histogram.html")

# Show the plot
show(p)


#Q5
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from bokeh.transform import linear_cmap
from bokeh.models import ColorBar
from bokeh.palettes import Viridis256

# Generate the data
np.random.seed(42)
data_heatmap = np.random.rand(10, 10)
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
xx, yy = np.meshgrid(x, y)
xx = xx.flatten()
yy = yy.flatten()
data = data_heatmap.flatten()

# Create a DataFrame
df = pd.DataFrame({
    'x': xx,
    'y': yy,
    'value': data
})

# Create the plot
p = figure(title="Heatmap", x_axis_label='X', y_axis_label='Y', 
           tools="hover,save,pan,box_zoom,wheel_zoom,reset",
           tooltips=[("X", "@x"), ("Y", "@y"), ("Value", "@value")],
           plot_width=600, plot_height=600)

# Create the color mapper
mapper = linear_cmap(field_name='value', palette=Viridis256, low=min(data), high=max(data))

# Add the heatmap to the plot
p.rect(x='x', y='y', width=0.1, height=0.1, source=df,
       line_color=None, fill_color=mapper)

# Add a color bar to the plot
color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
p.add_layout(color_bar, 'right')

# Specify the output file
output_file("heatmap.html")

# Show the plot
show(p)
