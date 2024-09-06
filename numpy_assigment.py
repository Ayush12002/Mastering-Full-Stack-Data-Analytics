import numpy as np

# Q1

# Create the NumPy array
arr = np.arange(6)

# Print the array
print("Array:", arr)

# Print the data type of the array
print("Data type:", arr.dtype)


# Q2

import numpy as np

# Assuming 'arr' is your NumPy array
arr = np.array([1.5, 2.6, 3.7])  # Example array, change this as needed

# Check if the data type of 'arr' is float64
is_float64 = arr.dtype == np.float64

# Print the result
print("Is the data type float64?", is_float64)


# Q3
import numpy as np

# Create the NumPy array with complex128 data type
arr = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)

# Print the array
print("Array:", arr)

# Print the data type of the array
print("Data type:", arr.dtype)


# Q4
import numpy as np

# Example NumPy array of integers
arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)

# Convert the array to float32 data type
arr_float32 = arr.astype(np.float32)

# Print the converted array
print("Converted Array:", arr_float32)

# Print the data type of the converted array
print("Data type:", arr_float32.dtype)

#Q5
import numpy as np

# Example NumPy array with float64 data type
arr = np.array([1.123456789, 2.987654321, 3.456789012], dtype=np.float64)

# Convert the array to float32 data type
arr_float32 = arr.astype(np.float32)

# Print the converted array
print("Converted Array:", arr_float32)

# Print the data type of the converted array
print("Data type:", arr_float32.dtype)


#Q6
import numpy as np

def array_attributes(arr):
    """
    Returns the shape, size, and data type of the given NumPy array.

    Parameters:
    arr (np.ndarray): The input NumPy array.

    Returns:
    tuple: A tuple containing the shape, size, and data type of the array.
    """
    shape = arr.shape
    size = arr.size
    dtype = arr.dtype
    
    return shape, size, dtype

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
shape, size, dtype = array_attributes(arr)

print("Shape:", shape)
print("Size:", size)
print("Data type:", dtype)

#Q7
import numpy as np

def array_dimension(arr):
    """
    Returns the dimensionality of the given NumPy array.

    Parameters:
    arr (np.ndarray): The input NumPy array.

    Returns:
    int: The number of dimensions of the array.
    """
    return arr.ndim

arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2-dimensional array
dimensionality = array_dimension(arr)

print("Dimensionality:", dimensionality)
#Q8
import numpy as np

def item_size_info(arr):
    """
    Returns the item size and the total size in bytes of the given NumPy array.

    Parameters:
    arr (np.ndarray): The input NumPy array.

    Returns:
    tuple: A tuple containing the item size (in bytes) and the total size (in bytes) of the array.
    """
    item_size = arr.itemsize
    total_size = arr.nbytes
    
    return item_size, total_size


arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # Array with float32 data type
item_size, total_size = item_size_info(arr)

print("Item size (in bytes):", item_size)
print("Total size (in bytes):", total_size)
#Q9
import numpy as np

def array_strides(arr):
    """
    Returns the strides of the given NumPy array.

    Parameters:
    arr (np.ndarray): The input NumPy array.

    Returns:
    tuple: A tuple containing the strides (in bytes) for each dimension of the array.
    """
    return arr.strides

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # Example array
strides = array_strides(arr)

print("Strides (in bytes):", strides)
#Q10
import numpy as np

def shape_stride_relationship(arr):
    """
    Returns the shape and strides of the given NumPy array.

    Parameters:
    arr (np.ndarray): The input NumPy array.

    Returns:
    tuple: A tuple containing two elements:
           - The shape of the array (as a tuple).
           - The strides of the array (as a tuple).
    """
    shape = arr.shape
    strides = arr.strides
    
    return shape, strides

# Example usage
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)  # Example array
shape, strides = shape_stride_relationship(arr)

print("Shape:", shape)
print("Strides (in bytes):", strides)
#Q11
import numpy as np

def create_zeros_array(n):
    """
    Creates a NumPy array of zeros with 'n' elements.

    Parameters:
    n (int): The number of elements in the array.

    Returns:
    np.ndarray: A NumPy array of zeros with 'n' elements.
    """
    if n < 0:
        raise ValueError("The number of elements must be a non-negative integer.")
    
    return np.zeros(n, dtype=np.float64)  # Default data type is float64, but you can change it if needed.

# Example usage
n = 5
zeros_array = create_zeros_array(n)

print("Array of zeros:", zeros_array)
#Q12
import numpy as np

def create_ones_matrix(rows, cols):
    """
    Creates a 2D NumPy array of ones with the specified number of rows and columns.

    Parameters:
    rows (int): The number of rows in the array.
    cols (int): The number of columns in the array.

    Returns:
    np.ndarray: A 2D NumPy array of ones with shape (rows, cols).
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("The number of rows and columns must be positive integers.")
    
    return np.ones((rows, cols), dtype=np.float64)  # Default data type is float64, but you can change it if needed.

# Example usage
rows = 3
cols = 4
ones_matrix = create_ones_matrix(rows, cols)

print("Matrix of ones:\n", ones_matrix)
#Q13
import numpy as np

def generate_range_array(start, stop, step):
    """
    Creates a NumPy array with a range starting from 'start', ending at 'stop' (exclusive), 
    and with the specified 'step'.

    Parameters:
    start (int): The starting value of the range.
    stop (int): The end value of the range (exclusive).
    step (int): The step size between consecutive values.

    Returns:
    np.ndarray: A NumPy array containing the range of values.
    """
    if step <= 0:
        raise ValueError("Step must be a positive integer.")
    
    return np.arange(start, stop, step)

# Example usage
start = 0
stop = 10
step = 2
range_array = generate_range_array(start, stop, step)

print("Range array:", range_array)
#Q14
import numpy as np

def generate_linear_space(start, stop, num):
    """
    Creates a NumPy array with 'num' equally spaced values between 'start' and 'stop' (inclusive).

    Parameters:
    start (float): The starting value of the range.
    stop (float): The ending value of the range (inclusive).
    num (int): The number of equally spaced values to generate.

    Returns:
    np.ndarray: A NumPy array containing the equally spaced values.
    """
    if num <= 0:
        raise ValueError("The number of values 'num' must be a positive integer.")
    
    return np.linspace(start, stop, num)

# Example usage
start = 1.0
stop = 5.0
num = 5
linear_space_array = generate_linear_space(start, stop, num)

print("Linear space array:", linear_space_array)
#Q15
import numpy as np

def create_identity_matrix(n):
    """
    Creates an identity matrix of size n x n.

    Parameters:
    n (int): The size of the identity matrix (number of rows and columns).

    Returns:
    np.ndarray: A square identity matrix of size n x n.
    """
    if n <= 0:
        raise ValueError("The size of the matrix 'n' must be a positive integer.")
    
    return np.eye(n, dtype=np.float64)  # Default data type is float64, but you can change it if needed.

# Example usage
n = 4
identity_matrix = create_identity_matrix(n)

print("Identity matrix:\n", identity_matrix)
#Q16
import numpy as np

def list_to_numpy_array(py_list):
    """
    Converts a Python list into a NumPy array.

    Parameters:
    py_list (list): The input Python list to be converted.

    Returns:
    np.ndarray: The NumPy array created from the input list.
    """
    # Convert the Python list to a NumPy array
    return np.array(py_list)

# Example usage
py_list = [1, 2, 3, 4, 5]
numpy_array = list_to_numpy_array(py_list)

print("NumPy array:", numpy_array)
print("Type:", type(numpy_array))
#Q17
import numpy as np

# Create an initial NumPy array
original_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

# Use numpy.view to create a new view of the original array
view_array = original_array.view()

# Modify the view_array
view_array[0, 0] = 10

# Print the original array and the view array to see the changes
print("Original array:\n", original_array)
print("View array:\n", view_array)

# Verify that both arrays share the same data
print("Original array data:", original_array.data)
print("View array data:", view_array.data)

# Check if both arrays share the same data
print("Do they share the same data?", original_array.data is view_array.data)
#Q18
import numpy as np

def concatenate_arrays(array1, array2, axis=0):
    """
    Concatenates two NumPy arrays along the specified axis.

    Parameters:
    array1 (np.ndarray): The first input array.
    array2 (np.ndarray): The second input array.
    axis (int): The axis along which to concatenate the arrays (default is 0).

    Returns:
    np.ndarray: The concatenated array.
    """
    # Check if the arrays have the same shape along the concatenation axis
    if array1.shape[axis] != array2.shape[axis]:
        raise ValueError("The dimensions of the arrays along the specified axis must be the same.")

    # Concatenate the arrays along the specified axis
    return np.concatenate((array1, array2), axis=axis)

# Example usage
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])

# Concatenate along axis 0 (vertically)
concatenated_array_axis0 = concatenate_arrays(array1, array2, axis=0)
print("Concatenated along axis 0:\n", concatenated_array_axis0)

# Concatenate along axis 1 (horizontally)
concatenated_array_axis1 = concatenate_arrays(array1, array2, axis=1)
print("Concatenated along axis 1:\n", concatenated_array_axis1)
#Q19
import numpy as np

# Create two NumPy arrays with compatible shapes for horizontal concatenation
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8], [9, 10]])

# Concatenate the arrays horizontally (axis=1)
concatenated_array = np.concatenate((array1, array2), axis=1)

print("Array 1:\n", array1)
print("Array 2:\n", array2)
print("Concatenated array:\n", concatenated_array)
#Q20
import numpy as np

def vertical_stack(arrays_list):
    """
    Vertically stacks multiple NumPy arrays provided as a list.

    Parameters:
    arrays_list (list of np.ndarray): A list of NumPy arrays to be stacked vertically.

    Returns:
    np.ndarray: The vertically stacked NumPy array.
    """
    if not all(isinstance(arr, np.ndarray) for arr in arrays_list):
        raise ValueError("All elements in the list must be NumPy arrays.")
    
    return np.vstack(arrays_list)

# Example usage
array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])
array3 = np.array([[13, 14, 15]])

# List of arrays to be vertically stacked
arrays_list = [array1, array2, array3]

# Stack the arrays vertically
stacked_array = vertical_stack(arrays_list)

print("Stacked array:\n", stacked_array)
#Q21
import numpy as np

def create_range_array(start, stop, step):
    """
    Creates a NumPy array of integers within a specified range (inclusive) with a given step size.

    Parameters:
    start (int): The starting value of the range.
    stop (int): The ending value of the range (inclusive).
    step (int): The step size between consecutive values.

    Returns:
    np.ndarray: A NumPy array of integers within the specified range.
    """
    if step <= 0:
        raise ValueError("Step size must be a positive integer.")
    
    # Adjust the stop value to be inclusive
    adjusted_stop = stop + 1 if step > 0 else stop - 1
    
    return np.arange(start, adjusted_stop, step)

# Example usage
start = 0
stop = 10
step = 2
range_array = create_range_array(start, stop, step)

print("Range array:", range_array)
#Q22
import numpy as np

def generate_equal_spacing(start, stop, num):
    """
    Generates an array of `num` equally spaced values between `start` and `stop` (inclusive).

    Parameters:
    start (float): The starting value of the range.
    stop (float): The ending value of the range (inclusive).
    num (int): The number of equally spaced values to generate.

    Returns:
    np.ndarray: A NumPy array containing the equally spaced values.
    """
    if num <= 0:
        raise ValueError("The number of values 'num' must be a positive integer.")
    
    return np.linspace(start, stop, num)

# Example usage
start = 0
stop = 1
num = 10
spacing_array = generate_equal_spacing(start, stop, num)

print("Array of equally spaced values:", spacing_array)
#Q23
import numpy as np

def generate_logarithmic_spacing(start, stop, num):
    """
    Generates an array of `num` logarithmically spaced values between `start` and `stop` (inclusive).

    Parameters:
    start (float): The starting value of the range (base-10 exponent).
    stop (float): The ending value of the range (base-10 exponent).
    num (int): The number of logarithmically spaced values to generate.

    Returns:
    np.ndarray: A NumPy array containing the logarithmically spaced values.
    """
    if num <= 0:
        raise ValueError("The number of values 'num' must be a positive integer.")
    
    return np.logspace(np.log10(start), np.log10(stop), num)

# Example usage
start = 1
stop = 1000
num = 5
log_spacing_array = generate_logarithmic_spacing(start, stop, num)

print("Array of logarithmically spaced values:", log_spacing_array)
#Q24
import numpy as np
import pandas as pd

def create_dataframe_from_array(rows, cols, low, high):
    """
    Creates a Pandas DataFrame using a NumPy array with random integers.

    Parameters:
    rows (int): The number of rows in the DataFrame.
    cols (int): The number of columns in the DataFrame.
    low (int): The lower bound of the random integers (inclusive).
    high (int): The upper bound of the random integers (exclusive).

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the random integers.
    """
    # Generate a NumPy array with random integers
    data = np.random.randint(low, high, size=(rows, cols))
    
    # Create a DataFrame from the NumPy array
    df = pd.DataFrame(data, columns=[f'Column_{i+1}' for i in range(cols)])
    
    return df

# Example usage
rows = 5
cols = 3
low = 1
high = 100
df = create_dataframe_from_array(rows, cols, low, high)

print("Pandas DataFrame:\n", df)
#Q25
import pandas as pd
import numpy as np

def replace_negatives_with_zeros(df, column_name):
    """
    Replaces all negative values in the specified column of the DataFrame with zeros.

    Parameters:
    df (pd.DataFrame): The DataFrame in which to replace negative values.
    column_name (str): The name of the column in which to replace negative values.

    Returns:
    pd.DataFrame: The DataFrame with negative values replaced by zeros in the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Use NumPy to replace negative values with zeros
    df[column_name] = np.where(df[column_name] < 0, 0, df[column_name])
    
    return df

# Example usage
data = {
    'A': [1, -2, 3, -4, 5],
    'B': [-1, 2, -3, 4, -5]
}
df = pd.DataFrame(data)

# Replace negative values in column 'A'
df_modified = replace_negatives_with_zeros(df, 'A')

print("Modified DataFrame:\n", df_modified)
#Q26
import numpy as np

# Create a NumPy array
array = np.array([10, 20, 30, 40, 50])

# Access the 3rd element (index 2)
third_element = array[2]

print("The 3rd element is:", third_element)

#Q27
import numpy as np

# Create a 2D NumPy array
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# Access the element at index (1, 2)
element = array_2d[1, 2]

print("Element at index (1, 2):", element)
#Q28
import numpy as np

# Create a NumPy array
array = np.array([3, 8, 2, 10, 5,7])

# Create a boolean mask for elements greater than 5
mask = array > 5

# Use the boolean mask to extract elements greater than 5
elements_greater_than_5 = array[mask]

print("Elements greater than 5:", elements_greater_than_5)
#Q29
import numpy as np

# Create a NumPy array
array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Slice the array from index 2 to 5 (inclusive)
sliced_array = array[2:6]  # Note: end index is exclusive, so use 6 to include index 5

print("Sliced array:", sliced_array)
#Q30
import numpy as np

# Create a 2D NumPy array
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

# Slice the array to extract the sub-array [[2, 3], [5, 6]]
# The rows we need are 1 and 2, and the columns are 1 and 2
sub_array = array_2d[1:3, 1:3]

print("Extracted sub-array:\n", sub_array)
#Q31
import numpy as np

def extract_elements_by_indices(array_2d, indices):
    """
    Extract elements from a 2D NumPy array based on indices provided in another array.

    Parameters:
    array_2d (np.ndarray): The 2D array from which to extract elements.
    indices (np.ndarray): An array of indices where each row is a pair of (row_index, column_index).

    Returns:
    np.ndarray: An array of extracted elements.
    """
    # Convert indices to lists of row and column indices
    row_indices, col_indices = indices[:, 0], indices[:, 1]
    
    # Use advanced indexing to extract the elements
    extracted_elements = array_2d[row_indices, col_indices]
    
    return extracted_elements

# Example usage
array_2d = np.array([[10, 20, 30],
                     [40, 50, 60],
                     [70, 80, 90]])

# Indices of elements to extract
indices = np.array([[0, 1],  # element at (0, 1) which is 20
                    [1, 2],  # element at (1, 2) which is 60
                    [2, 0]]) # element at (2, 0) which is 70

# Extract elements based on indices
extracted_elements = extract_elements_by_indices(array_2d, indices)

print("Extracted elements:", extracted_elements)
#Q32
import numpy as np

def filter_elements_above_threshold(array, threshold):
    """
    Filters elements from a 1D NumPy array that are greater than the given threshold.

    Parameters:
    array (np.ndarray): The 1D array to filter.
    threshold (float): The threshold value for filtering.

    Returns:
    np.ndarray: A 1D array with elements greater than the threshold.
    """
    # Create a boolean mask where elements are greater than the threshold
    mask = array > threshold
    
    # Use the boolean mask to filter the array
    filtered_array = array[mask]
    
    return filtered_array

# Example usage
array = np.array([1, 5, 8, 12, 3, 7])
threshold = 6

# Filter elements greater than the threshold
filtered_elements = filter_elements_above_threshold(array, threshold)

print("Filtered elements:", filtered_elements)
#Q33
import numpy as np

def extract_elements_from_3d_array(array_3d, x_indices, y_indices, z_indices):
    """
    Extract specific elements from a 3D NumPy array using indices provided in three separate arrays.

    Parameters:
    array_3d (np.ndarray): The 3D array from which to extract elements.
    x_indices (np.ndarray): An array of indices along the x-dimension.
    y_indices (np.ndarray): An array of indices along the y-dimension.
    z_indices (np.ndarray): An array of indices along the z-dimension.

    Returns:
    np.ndarray: An array of extracted elements.
    """
    # Convert indices to lists if they are not already
    x_indices, y_indices, z_indices = np.array(x_indices), np.array(y_indices), np.array(z_indices)
    
    # Use advanced indexing to extract the elements
    extracted_elements = array_3d[x_indices, y_indices, z_indices]
    
    return extracted_elements

# Example usage
array_3d = np.array([[[ 1,  2,  3],
                      [ 4,  5,  6],
                      [ 7,  8,  9]],
                     
                     [[10, 11, 12],
                      [13, 14, 15],
                      [16, 17, 18]],
                     
                     [[19, 20, 21],
                      [22, 23, 24],
                      [25, 26, 27]]])

# Indices for extraction
x_indices = np.array([0, 1, 2])  # Row indices
y_indices = np.array([1, 2, 0])  # Column indices
z_indices = np.array([2, 0, 1])  # Depth indices

# Extract elements based on indices
extracted_elements = extract_elements_from_3d_array(array_3d, x_indices, y_indices, z_indices)

print("Extracted elements:", extracted_elements)
#Q34
import numpy as np

def filter_elements_by_conditions(array, condition1, condition2):
    """
    Returns elements from a NumPy array where both conditions are satisfied.

    Parameters:
    array (np.ndarray): The array to filter.
    condition1 (np.ndarray): A boolean array or condition for filtering.
    condition2 (np.ndarray): A boolean array or condition for filtering.

    Returns:
    np.ndarray: An array of elements where both conditions are satisfied.
    """
    # Ensure conditions are boolean arrays
    condition1 = np.asarray(condition1)
    condition2 = np.asarray(condition2)
    
    # Combine conditions using logical AND
    combined_condition = condition1 & condition2
    
    # Filter the array using the combined condition
    filtered_elements = array[combined_condition]
    
    return filtered_elements

# Example usage
array = np.array([1, 4, 6, 8, 10, 12, 14])

# Define conditions
condition1 = array > 5      # Elements greater than 5
condition2 = array % 2 == 0 # Elements that are even

# Filter elements where both conditions are satisfied
filtered_elements = filter_elements_by_conditions(array, condition1, condition2)

print("Filtered elements:", filtered_elements)
#Q35
import numpy as np

def extract_elements_from_2d_array(array_2d, row_indices, col_indices):
    """
    Extract specific elements from a 2D NumPy array using row and column indices provided in separate arrays.

    Parameters:
    array_2d (np.ndarray): The 2D array from which to extract elements.
    row_indices (np.ndarray): An array of row indices.
    col_indices (np.ndarray): An array of column indices.

    Returns:
    np.ndarray: An array of extracted elements.
    """
    # Convert indices to arrays if they are not already
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)
    
    # Use advanced indexing to extract the elements
    extracted_elements = array_2d[row_indices, col_indices]
    
    return extracted_elements

# Example usage
array_2d = np.array([[10, 20, 30],
                     [40, 50, 60],
                     [70, 80, 90]])

# Row and column indices for extraction
row_indices = np.array([0, 1, 2])  # Row indices
col_indices = np.array([1, 2, 0])  # Column indices

# Extract elements based on indices
extracted_elements = extract_elements_from_2d_array(array_2d, row_indices, col_indices)

print("Extracted elements:", extracted_elements)
#Q36
import numpy as np

# Create a 2D NumPy array of shape (3, 3)
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Add a scalar value of 5 to each element using broadcasting
result = arr + 5

print("Original array:\n", arr)
print("Array after adding 5:\n", result)
#Q37
import numpy as np

# Define the arrays
arr1 = np.array([[2, 4, 6]])  # Shape (1, 3)
arr2 = np.array([[10, 20, 30, 40],
                 [50, 60, 70, 80],
                 [90, 100, 110, 120]])  # Shape (3, 4)

# Reshape arr1 to shape (3, 1) to enable broadcasting with arr2
arr1_reshaped = arr1.T  # Transpose arr1 to shape (3, 1)

# Multiply each row of arr2 by the corresponding element in arr1
result = arr2 * arr1_reshaped

print("arr1:\n", arr1)
print("arr2:\n", arr2)
print("Result of multiplication:\n", result)
#Q38
import numpy as np


arr1 = np.array([[1, 2, 3, 4]])  # Shape (1, 4)
arr2 = np.array([[5, 6, 7],
                 [8, 9, 10],
                 [11, 12, 13],
                 [14, 15, 16]])  # Shape (4, 3)


result = arr2 + arr1[:, :3]

print("Array arr1:\n", arr1)
print("Array arr2:\n", arr2)
print("Result of addition:\n", result)

#Q39
import numpy as np

# Create the arrays
arr1 = np.array([[3], [1]])  # Shape (3, 1)
arr2 = np.array([[1, 3]])      # Shape (1, 3)

# Add the arrays using broadcasting
result = arr1 + arr2

print("Array arr1:\n", arr1)
print("Array arr2:\n", arr2)
print("Result of addition:\n", result)
#Q40
import numpy as np

# Create the arrays
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])   # Shape (2, 3)
arr2 = np.array([[10, 20],
                 [30, 40]])   # Shape (2, 2)

# Reshape arr2 to be compatible for broadcasting with arr1
# Adding an extra dimension to arr2
arr2_reshaped = arr2[:, :, np.newaxis]  # Shape (2, 2, 1)

# Reshape arr1 to be compatible with arr2_reshaped
arr1_reshaped = arr1[:, np.newaxis, :]  # Shape (2, 1, 3)

# Perform multiplication with broadcasting
result = arr1_reshaped * arr2_reshaped

print("Array arr1:\n", arr1)
print("Array arr2:\n", arr2)
print("Array arr2 reshaped:\n", arr2_reshaped)
print("Array arr1 reshaped:\n", arr1_reshaped)
print("Result of multiplication:\n", result)

#Q41
import numpy as np

# Create a 2D NumPy array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Calculate the column-wise mean
column_mean = np.mean(arr, axis=0)

print("Original array:\n", arr)
print("Column-wise mean:", column_mean)
#Q42
import numpy as np

# Create a 2D NumPy array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Find the maximum value in each row
row_max = np.max(arr, axis=1)

print("Original array:\n", arr)
print("Maximum value in each row:", row_max)
#Q43
import numpy as np

# Create a 2D NumPy array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Find the indices of the maximum values in each column
indices_of_max = np.argmax(arr, axis=0)

print("Original array:\n", arr)
print("Indices of maximum values in each column:", indices_of_max)
#Q44
import numpy as np

def moving_sum(arr, window_size):
    """
    Compute the moving sum along rows of a 2D array with a given window size.
    
    Parameters:
    - arr: 2D NumPy array
    - window_size: size of the moving window
    
    Returns:
    - 2D NumPy array with the moving sums
    """
    # Ensure the window size is valid
    if window_size <= 0 or window_size > arr.shape[1]:
        raise ValueError("Window size must be greater than 0 and less than or equal to the number of columns")

    # Initialize an array to store the result
    result = np.zeros((arr.shape[0], arr.shape[1] - window_size + 1))
    
    # Compute the moving sum for each row
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1] - window_size + 1):
            result[i, j] = np.sum(arr[i, j:j + window_size])
    
    return result

# Example usage
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

window_size = 3
moving_sum_result = moving_sum(arr, window_size)

print("Original array:\n", arr)
print("Moving sum with window size", window_size, ":\n", moving_sum_result)
#Q45
import numpy as np

def are_all_elements_even(arr):
    """
    Check if all elements in each column of a 2D array are even.
    
    Parameters:
    - arr: 2D NumPy array
    
    Returns:
    - A boolean array where each value represents if all elements in the corresponding column are even.
    """
    # Check if elements are even
    even_check = arr % 2 == 0
    
    # Check if all elements in each column are even
    all_even = np.all(even_check, axis=0)
    
    return all_even

# Example usage
arr = np.array([[2, 4, 6],
                [3, 5, 7]])

result = are_all_elements_even(arr)

print("Original array:\n", arr)
print("All elements in each column are even:", result)
#Q46
import numpy as np

# Define the original array
original_array = np.array([1, 2, 3, 4, 5, 6])

# Desired dimensions
m, n = 2, 3

# Reshape the array
reshaped_matrix = original_array.reshape((m, n))

print("Original array:\n", original_array)
print("Reshaped matrix ({}x{}):\n".format(m, n), reshaped_matrix)
#Q47
import numpy as np

def flatten_matrix(matrix):
    """
    Flatten a 2D NumPy matrix into a 1D array.
    
    Parameters:
    - matrix: 2D NumPy array to flatten
    
    Returns:
    - 1D NumPy array that is a flattened version of the input matrix
    """
    return matrix.flatten()  # Alternatively, you can use matrix.ravel()

# Example usage
input_matrix = np.array([[1, 2, 3], [4, 5, 6]])
flattened_array = flatten_matrix(input_matrix)

print("Input matrix:\n", input_matrix)
print("Flattened array:\n", flattened_array)
#Q48
import numpy as np

def concatenate_arrays(array1, array2, axis):
    """
    Concatenate two NumPy arrays along a specified axis.
    
    Parameters:
    - array1: First NumPy array
    - array2: Second NumPy array
    - axis: Axis along which to concatenate (0 for rows, 1 for columns)
    
    Returns:
    - Concatenated NumPy array
    """
    # Check if the arrays can be concatenated along the specified axis
    if axis == 0 and array1.shape[1] != array2.shape[1]:
        raise ValueError("For axis=0, the number of columns must be the same.")
    if axis == 1 and array1.shape[0] != array2.shape[0]:
        raise ValueError("For axis=1, the number of rows must be the same.")
    
    # Concatenate the arrays
    concatenated_array = np.concatenate((array1, array2), axis=axis)
    return concatenated_array

# Example usage
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

# Concatenate along axis 0 (rows)
result_axis0 = concatenate_arrays(array1, array2, axis=0)
print("Concatenated along axis 0:\n", result_axis0)

# Concatenate along axis 1 (columns)
result_axis1 = concatenate_arrays(array1, array2, axis=1)
print("Concatenated along axis 1:\n", result_axis1)
#Q49
import numpy as np

def split_array(array, axis, indices_or_sections):
    """
    Split a NumPy array into multiple sub-arrays along a specified axis.
    
    Parameters:
    - array: NumPy array to split
    - axis: Axis along which to split (0 for rows, 1 for columns)
    - indices_or_sections: If an integer, it specifies the number of equal sections to split the array into.
                            If a list of integers, it specifies the indices at which to split.
    
    Returns:
    - List of NumPy arrays (sub-arrays)
    """
    # Split the array
    split_arrays = np.split(array, indices_or_sections, axis=axis)
    return split_arrays

# Example usage
original_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Split along axis 0 (rows) into 3 equal parts
split_axis0 = split_array(original_array, axis=0, indices_or_sections=3)
print("Split along axis 0:\n", split_axis0)

# Split along axis 1 (columns) into 3 equal parts
split_axis1 = split_array(original_array, axis=1, indices_or_sections=3)
print("Split along axis 1:\n", split_axis1)
#Q50
import numpy as np

def insert_and_delete_elements(array, indices_to_insert, values_to_insert, indices_to_delete):
    """
    Insert and then delete elements in a NumPy array at specified indices.
    
    Parameters:
    - array: Original NumPy array
    - indices_to_insert: List of indices where new elements should be inserted
    - values_to_insert: List of values to insert at the specified indices
    - indices_to_delete: List of indices of elements to delete
    
    Returns:
    - Modified NumPy array after insertion and deletion
    """
    # Convert lists to NumPy arrays for compatibility with numpy functions
    indices_to_insert = np.array(indices_to_insert)
    values_to_insert = np.array(values_to_insert)
    
    # Insert elements
    new_array = np.insert(array, indices_to_insert, values_to_insert)
    
    # Delete elements
    new_array = np.delete(new_array, indices_to_delete)
    
    return new_array

# Example usage
original_array = np.array([1, 2, 3, 4, 5])
indices_to_insert = [2, 4]
values_to_insert = [10, 11]
indices_to_delete = [1, 3]

modified_array = insert_and_delete_elements(original_array, indices_to_insert, values_to_insert, indices_to_delete)

print("Original array:\n", original_array)
print("Modified array:\n", modified_array)
#Q51
import numpy as np

# Create arr1 with random integers
arr1 = np.random.randint(1, 10, size=(3, 3))  # Example: 3x3 array with random integers between 1 and 10

# Create arr2 with integers from 1 to 10
arr2 = np.arange(1, 10).reshape((3, 3))  # Example: 3x3 array with integers from 1 to 9

# Perform element-wise addition
result = arr1 + arr2

print("Array 1:\n", arr1)
print("Array 2:\n", arr2)
print("Element-wise addition result:\n", result)
#Q52
import numpy as np

# Create arr1 with sequential integers from 10 to 1
arr1 = np.arange(10, 0, -1)  # 10 to 1, inclusive

# Create arr2 with integers from 1 to 10
arr2 = np.arange(1, 11)  # 1 to 10, inclusive

# Ensure both arrays have the same shape for element-wise operations
# Reshape both arrays to 1D arrays with the same length
arr1 = arr1.reshape(1, -1)  # Reshape to a 1D row vector
arr2 = arr2.reshape(1, -1)  # Reshape to a 1D row vector

# Perform element-wise subtraction
result = arr1 - arr2

print("Array 1:\n", arr1)
print("Array 2:\n", arr2)
print("Element-wise subtraction result:\n", result)
#Q53
import numpy as np

# Create arr1 with random integers (for example, 2x5 array with random integers between 1 and 10)
arr1 = np.random.randint(1, 10, size=(2, 5))

# Create arr2 with integers from 1 to 5
arr2 = np.arange(1, 6).reshape(1, -1)  # 1x5 array to match the shape of arr1

# Ensure arr2 has the same shape as arr1 for element-wise multiplication
arr2 = np.tile(arr2, (arr1.shape[0], 1))  # Repeat arr2 to match the shape of arr1

# Perform element-wise multiplication
result = arr1 * arr2

print("Array 1:\n", arr1)
print("Array 2:\n", arr2)
print("Element-wise multiplication result:\n", result)
#Q54
import numpy as np

# Create arr1 with even integers from 2 to 10
arr1 = np.arange(2, 11, 2)  # Output: [ 2  4  6  8 10]

# Create arr2 with integers from 1 to 5
arr2 = np.arange(1, 6)  # Output: [1 2 3 4 5]

# Ensure both arrays have the same shape for element-wise division
# Reshape arr2 to match the shape of arr1 if necessary
arr1 = arr1.reshape(1, -1)  # Reshape to a 1x5 array
arr2 = arr2.reshape(1, -1)  # Reshape to a 1x5 array

# Perform element-wise division
result = arr1 / arr2

print("Array 1:\n", arr1)
print("Array 2:\n", arr2)
print("Element-wise division result:\n", result)
#Q55
import numpy as np

# Create arr1 with integers from 1 to 5
arr1 = np.arange(1, 6)  # Output: [1, 2, 3, 4, 5]

# Create arr2 with the same numbers reversed
arr2 = np.arange(1, 6)[::-1]  # Output: [5, 4, 3, 2, 1]

# Perform element-wise exponentiation
result = arr1 ** arr2

print("Array 1:\n", arr1)
print("Array 2:\n", arr2)
print("Element-wise exponentiation result:\n", result)
#Q56
import numpy as np

def count_substring_occurrences(array, substring):
    """
    Counts the occurrences of a specific substring within a NumPy array of strings.

    Parameters:
    - array (np.ndarray): A NumPy array containing strings.
    - substring (str): The substring to search for.

    Returns:
    - int: The total count of the substring occurrences.
    """
    # Initialize count
    total_count = 0

    # Iterate through each string in the array
    for string in array:
        # Count occurrences of the substring in the current string
        total_count += string.count(substring)
    
    return total_count

# Define the NumPy array and substring
arr = np.array(['hello', 'world', 'hello', 'numpy', 'hello'])
substring_to_search = 'hello'

# Call the function
occurrences = count_substring_occurrences(arr, substring_to_search)
print(f"Total occurrences of '{substring_to_search}':", occurrences)
#Q57
import numpy as np

def extract_uppercase_characters(array):
    """
    Extracts uppercase characters from a NumPy array of strings.

    Parameters:
    - array (np.ndarray): A NumPy array containing strings.

    Returns:
    - str: A string containing all uppercase characters found in the array.
    """
    uppercase_chars = []

    # Iterate through each string in the array
    for string in array:
        # Filter out uppercase characters
        uppercase_chars.extend([char for char in string if char.isupper()])
    
    # Join list of characters into a single string
    return ''.join(uppercase_chars)

# Define the NumPy array
arr = np.array(['Hello', 'World', 'OpenAI', 'GPT'])

# Call the function
uppercase_result = extract_uppercase_characters(arr)
print("Extracted uppercase characters:", uppercase_result)
#Q58
import numpy as np

def replace_substring(array, old_substring, new_string):
    """
    Replaces occurrences of a substring in a NumPy array of strings with a new string.

    Parameters:
    - array (np.ndarray): A NumPy array containing strings.
    - old_substring (str): The substring to be replaced.
    - new_string (str): The new string to replace the old substring.

    Returns:
    - np.ndarray: A NumPy array with the updated strings.
    """
    # Use vectorized operations to replace the substring in each string of the array
    replaced_array = np.char.replace(array, old_substring, new_string)
    
    return replaced_array

# Define the NumPy array
arr = np.array(['apple', 'banana', 'grape', 'pineapple'])

# Define the old substring and new string
old_substring = 'apple'
new_string = 'orange'

# Call the function
updated_arr = replace_substring(arr, old_substring, new_string)
print("Updated array:\n", updated_arr)
#Q59
import numpy as np

def concatenate_strings_element_wise(arr1, arr2):
    """
    Concatenates strings in two NumPy arrays element-wise.

    Parameters:
    - arr1 (np.ndarray): A NumPy array containing the first set of strings.
    - arr2 (np.ndarray): A NumPy array containing the second set of strings.

    Returns:
    - np.ndarray: A NumPy array with concatenated strings.
    """
    # Ensure both arrays have the same shape
    if arr1.shape != arr2.shape:
        raise ValueError("Both arrays must have the same shape.")
    
    # Concatenate strings element-wise
    concatenated_array = np.char.add(arr1, arr2)
    
    return concatenated_array

# Define the NumPy arrays
arr1 = np.array(['Hello', 'World'])
arr2 = np.array(['Open', 'AI'])

# Call the function
result = concatenate_strings_element_wise(arr1, arr2)
print("Concatenated array:\n", result)
#Q60
import numpy as np

def length_of_longest_string(array):
    """
    Finds the length of the longest string in a NumPy array of strings.

    Parameters:
    - array (np.ndarray): A NumPy array containing strings.

    Returns:
    - int: The length of the longest string in the array.
    """
    # Compute the length of each string in the array
    lengths = np.char.str_len(array)
    
    # Find the maximum length
    max_length = np.max(lengths)
    
    return max_length

# Define the NumPy array
arr = np.array(['apple', 'banana', 'grape', 'pineapple'])

# Call the function
longest_length = length_of_longest_string(arr)
print("Length of the longest string:", longest_length)
#Q61
import numpy as np

# Step 1: Create a dataset of 100 random integers between 1 and 1000
dataset = np.random.randint(1, 1001, size=100)

# Step 2: Compute statistical measures
mean = np.mean(dataset)
median = np.median(dataset)
variance = np.var(dataset)
std_dev = np.std(dataset)

# Print the results
print(f"Dataset: {dataset}")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
#Q62
import numpy as np

# Step 1: Generate an array of 50 random numbers between 1 and 100
dataset = np.random.randint(1, 101, size=50)

# Step 2: Compute the 25th and 75th percentiles
percentile_25 = np.percentile(dataset, 25)
percentile_75 = np.percentile(dataset, 75)

# Print the results
print(f"Dataset: {dataset}")
print(f"25th Percentile: {percentile_25}")
print(f"75th Percentile: {percentile_75}")
#Q63
import numpy as np

# Step 1: Create two arrays representing two sets of variables
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([2, 3, 4, 5, 6])

# Step 2: Compute the correlation coefficient matrix
correlation_matrix = np.corrcoef(array1, array2)

# Extract the correlation coefficient between the two arrays
correlation_coefficient = correlation_matrix[0, 1]

# Print the results
print("Correlation matrix:\n", correlation_matrix)
print("Correlation coefficient between the two arrays:", correlation_coefficient)
#Q64
import numpy as np

# Step 1: Create two matrices
matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6]])

matrix2 = np.array([[7, 8],
                    [9, 10],
                    [11, 12]])

# Step 2: Perform matrix multiplication using np.dot
result = np.dot(matrix1, matrix2)

# Print the result
print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Result of matrix multiplication:\n", result)
#Q65
import numpy as np

# Step 1: Create an array of 50 random integers between 10 and 1000
data = np.random.randint(10, 1001, size=50)

# Step 2: Calculate percentiles
percentile_10 = np.percentile(data, 10)
percentile_50 = np.percentile(data, 50)  # Median
percentile_90 = np.percentile(data, 90)
first_quartile = np.percentile(data, 25)
third_quartile = np.percentile(data, 75)

# Print the results
print(f"Array:\n{data}")
print(f"10th Percentile: {percentile_10}")
print(f"50th Percentile (Median): {percentile_50}")
print(f"90th Percentile: {percentile_90}")
print(f"First Quartile (25th Percentile): {first_quartile}")
print(f"Third Quartile (75th Percentile): {third_quartile}")
#Q66
import numpy as np

# Step 1: Create a NumPy array of integers
array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Define the specific element to find
element_to_find = 50

# Step 2: Find the index of the specific element
indices = np.where(array == element_to_find)

# Convert indices to a list if you want a more readable output
indices_list = indices[0].tolist()

# Print the results
print(f"Array: {array}")
print(f"Element to find: {element_to_find}")
print(f"Indices of the element {element_to_find}: {indices_list}")
#Q67
import numpy as np

# Step 1: Generate a random NumPy array
# For example, create an array of 10 random integers between 1 and 100
random_array = np.random.randint(1, 101, size=10)

# Step 2: Sort the array in ascending order
sorted_array = np.sort(random_array)

# Print the results
print("Original array:\n", random_array)
print("Sorted array:\n", sorted_array)
#Q68
import numpy as np

# Given NumPy array
arr = np.array([12, 25, 6, 42, 8, 30])

# Create a boolean mask for elements greater than 20
mask = arr > 20

# Apply the mask to filter the array
filtered_elements = arr[mask]

# Print the results
print("Original array:", arr)
print("Filtered elements (greater than 20):", filtered_elements)
#Q69
import numpy as np

# Given NumPy array
arr = np.array([1, 5, 8, 12, 15])

# Create a boolean mask for elements divisible by 3
mask = arr % 3 == 0

# Apply the mask to filter the array
filtered_elements = arr[mask]

# Print the results
print("Original array:", arr)
print("Filtered elements (divisible by 3):", filtered_elements)
#Q70
import numpy as np

# Given NumPy array
arr = np.array([10, 20, 30, 40, 50])

# Create a boolean mask for elements between 20 and 40 (inclusive)
mask = (arr >= 20) & (arr <= 40)

# Apply the mask to filter the array
filtered_elements = arr[mask]

# Print the results
print("Original array:", arr)
print("Filtered elements (≥ 20 and ≤ 40):", filtered_elements)
#Q71
import numpy as np

# Given NumPy array
arr = np.array([1, 2, 3])

# Check the byte order
byte_order = arr.dtype.byteorder

# Print the results
print("Array:", arr)
print("Byte order:", byte_order)
#Q72
import numpy as np

# Step 1: Create the NumPy array with dtype=np.int32
arr = np.array([1, 2, 3], dtype=np.int32)

# Print the original array and its byte order
print("Original array:", arr)
print("Original byte order:", arr.dtype.byteorder)

# Step 2: Perform byte swapping in place
arr.byteswap(inplace=True)

# Print the modified array and its byte order
print("Array after byteswap:", arr)
print("Byte order after byteswap:", arr.dtype.byteorder)
#Q73
import numpy as np

# Create the NumPy array with dtype=np.int32
arr = np.array([1, 2, 3], dtype=np.int32)

# Print the original array and its byte order
print("Original array:", arr)
print("Original byte order:", arr.dtype.byteorder)

# Swap the byte order without modifying the original array
swapped_arr = arr.newbyteorder()

# Print the new array and its byte order
print("Array with swapped byte order:", swapped_arr)
print("Byte order of new array:", swapped_arr.dtype.byteorder)
#Q74
import numpy as np

# Create the NumPy array with dtype=np.int32
arr = np.array([1, 2, 3], dtype=np.int32)

# Print the original array and its byte order
print("Original array:", arr)
print("Original byte order:", arr.dtype.byteorder)

# Determine system's native byte order
system_byte_order = '<' if np.little_endian else '>'

# Swap byte order conditionally based on system's native endianness
if arr.dtype.byteorder == system_byte_order:
    swapped_arr = arr.newbyteorder('>')
else:
    swapped_arr = arr.newbyteorder('<')

# Print the new array and its byte order
print("Array with swapped byte order:", swapped_arr)
print("Byte order of new array:", swapped_arr.dtype.byteorder)
#Q75
import numpy as np

# Create the NumPy array with dtype=np.int32
rr = np.array([1, 2, 3], dtype=np.int32)

# Determine the system's native byte order
native_byte_order = '<' if np.little_endian else '>'

# Get the byte order of the array's dtype
array_byte_order = rr.dtype.byteorder

# Check if byte swapping is necessary
if array_byte_order == native_byte_order or array_byte_order == '=':
    print("Byte swapping is not necessary.")
else:
    print("Byte swapping is necessary.")

# Print the byte orders for reference
print("System's native byte order:", native_byte_order)
print("Array's byte order:", array_byte_order)
#Q76
import numpy as np

# Step 1: Create the original NumPy array
arr1 = np.arange(1, 11)  # Array with values from 1 to 10

# Step 2: Create a copy of the original array
copy_arr = arr1.copy()

# Step 3: Modify an element in the copy
copy_arr[0] = 100

# Step 4: Check if modifying the copy affects the original array
print("Original array (arr1):", arr1)
print("Modified copy (copy_arr):", copy_arr)
#Q77
import numpy as np

# Step 1: Create the 2D NumPy array with random integers
np.random.seed(0)  # For reproducibility
matrix = np.random.randint(1, 10, size=(3, 3))

# Step 2: Extract a slice from the matrix
view_slice = matrix[1:3, 0:2]  # Extracting a sub-array (2x2)

# Step 3: Modify an element in the slice
view_slice[0, 0] = 99

# Step 4: Check if modifying the slice affects the original matrix
print("Original matrix:")
print(matrix)

print("View slice:")
print(view_slice)
#Q78
import numpy as np

# Step 1: Create the NumPy array with sequential integers
array_a = np.arange(1, 13).reshape(4, 3)

# Step 2: Extract a slice from the array
view_b = array_a[1:3, 0:2]  # Extracting a sub-array (2x2)

# Step 3: Broadcast addition of 5 to the slice
view_b += 5

# Step 4: Check if modifying the slice affects the original array
print("Original array (array_a):")
print(array_a)

print("Modified slice (view_b):")
print(view_b)
#Q79
import numpy as np

# Step 1: Create the original NumPy array with values from 1 to 8
orig_array = np.arange(1, 9).reshape(2, 4)

# Step 2: Create a reshaped view of shape (4, 2)
reshaped_view = orig_array.reshape(4, 2)

# Step 3: Modify an element in the reshaped view
reshaped_view[1, 1] = 99

# Step 4: Check if modifying the view affects the original array
print("Original array (orig_array):")
print(orig_array)

print("Reshaped view (reshaped_view):")
print(reshaped_view)
#Q80
import numpy as np

# Step 1: Create the original NumPy array with random integers
np.random.seed(0)  # For reproducibility
data = np.random.randint(1, 10, size=(3, 4))

# Step 2: Extract a copy of elements greater than 5
data_copy = data[data > 5].copy()  # Using .copy() to ensure we have a separate copy

# Step 3: Modify an element in the copy
data_copy[0] = 99  # Modify the first element in the copy

# Step 4: Check if modifying the copy affects the original array
print("Original array (data):")
print(data)

print("Modified copy (data_copy):")
print(data_copy)
#Q81
import numpy as np

# Step 1: Create two matrices of identical shape with integer values
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# Step 2: Perform addition
addition_result = A + B

# Step 3: Perform subtraction
subtraction_result = A - B

# Step 4: Print results
print("Matrix A:")
print(A)

print("Matrix B:")
print(B)

print("Addition result (A + B):")
print(addition_result)

print("Subtraction result (A - B):")
print(subtraction_result)
#Q82
import numpy as np

# Step 1: Generate matrices C and D
C = np.random.randint(1, 10, size=(3, 2))  # 3x2 matrix with random integers
D = np.random.randint(1, 10, size=(2, 4))  # 2x4 matrix with random integers

# Step 2: Perform matrix multiplication
result = np.dot(C, D)  # Alternatively, you could use C @ D

# Step 3: Print results
print("Matrix C:")
print(C)

print("Matrix D:")
print(D)

print("Matrix multiplication result (C @ D):")
print(result)
#Q83
import numpy as np

# Step 1: Create the matrix E
E = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Step 2: Find the transpose of matrix E
E_transpose = E.T  # Alternatively, you could use np.transpose(E)

# Step 3: Print results
print("Matrix E:")
print(E)

print("Transpose of Matrix E:")
print(E_transpose)
#Q84
import numpy as np

# Step 1: Generate a square matrix F
np.random.seed(0)  # For reproducibility
F = np.random.randint(1, 10, size=(4, 4))  # A 4x4 matrix with random integers between 1 and 10

# Step 2: Compute the determinant of matrix F
determinant = np.linalg.det(F)

# Step 3: Print results
print("Matrix F:")
print(F)

print("Determinant of Matrix F:")
print(determinant)
#Q85
import numpy as np

# Step 1: Create a square matrix G
np.random.seed(0)  # For reproducibility
G = np.random.randint(1, 10, size=(3, 3))  # A 3x3 matrix with random integers between 1 and 10

# Check if the matrix is singular (i.e., has no inverse) by computing its determinant
determinant = np.linalg.det(G)

# Step 2: Compute the inverse of matrix G, if it is not singular
if determinant != 0:
    G_inverse = np.linalg.inv(G)
else:
    G_inverse = "Matrix is singular and cannot be inverted."

# Step 3: Print results
print("Matrix G:")
print(G)

print("Determinant of Matrix G:")
print(determinant)

print("Inverse of Matrix G:")
print(G_inverse)
