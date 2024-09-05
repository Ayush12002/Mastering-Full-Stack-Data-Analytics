# Q1 Write a code to reverse a string?

def reverse_string(s):
    return s[::-1]

s = "hello"
print(reverse_string(s))  # Output: "olleh"

# Q2 rite a code to count the number of vowels in a string?

def count_vowels(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)

s = "hello"
print(count_vowels(s)) 


# Q3 Write a code to check if a given string is a palindrome or not?

def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

s = "racecar"
print(is_palindrome(s)) 


# Q4 write a code to check if two given strings are anagrams of each other?

def are_anagrams(s1, s2):
    return sorted(s1) == sorted(s2)

s1 = "listen"
s2 = "silent"
print(are_anagrams(s1, s2))

# Q5 Write a code to find all occurrences of a given substring within another string?

def find_occurrences(s, substring):
    return [i for i in range(len(s)) if s.startswith(substring, i)]

s = "ababab"
substring = "ab"
print(find_occurrences(s, substring))

# Q6 Write a code to perform basic string compression using the counts of repeated characters?

def compress_string(s):
    compressed = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            compressed.append(s[i - 1] + str(count))
            count = 1
    compressed.append(s[-1] + str(count))
    return ''.join(compressed)

s = "aaabbbcccaaa"
print(compress_string(s))

# Q7 Write a code to determine if a string has all unique characters?

def has_unique_characters(s):
    return len(set(s)) == len(s)

s = "abcde"
print(has_unique_characters(s))

# Q8 write a code to convert a given string to uppercase or lowercase?

def convert_case(s, to_upper=True):
    return s.upper() if to_upper else s.lower()

s = "Hello World"
print(convert_case(s))  
print(convert_case(s, to_upper=False))

# Q9 write a code to count the number of words in a string?

def count_words(s):
    return len(s.split())

s = "Hello world"
print(count_words(s))

# Q10 write a code to concatenate two strings without using the + operator?

def concatenate_strings(s1, s2):
    return ''.join([s1, s2])

s1 = "Hello"
s2 = "World"
print(concatenate_strings(s1, s2))

# Q11 Write a code to remove all occurrences of a specific element from a list?

def remove_element(lst, element):
    return [x for x in lst if x != element]

lst = [1, 2, 3, 2, 4]
element = 2
print(remove_element(lst, element))

# Q12 Implement a code to find the second largest number in a given list of integers?

def second_largest(lst):
    first, second = float('-inf'), float('-inf')
    for number in lst:
        if number > first:
            second = first
            first = number
        elif first > number > second:
            second = number
    return second

lst = [10, 20, 4, 45, 99]
print(second_largest(lst))

# Q13 Create a code to count the occurrences of each element in a list and return a dictionary with elements askeys and their counts as values?

def count_occurrences(lst):
    from collections import Counter
    return dict(Counter(lst))

lst = [1, 2, 2, 3, 3, 3]
print(count_occurrences(lst)) 


# Q14 write a code to reverse a list in-place without using any built-in reverse functions?

def reverse_list(lst):
    left, right = 0, len(lst) - 1
    while left < right:
        lst[left], lst[right] = lst[right], lst[left]
        left += 1
        right -= 1


lst = [1, 2, 3, 4]
reverse_list(lst)
print(lst) 


# Q15 mplement a code to find and remove duplicates from a list while preserving the original order ofelements?

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

lst = [1, 2, 2, 3, 4, 4, 5]
print(remove_duplicates(lst))

# Q16 Create a code to check if a given list is sorted (either in ascending or descending order) or not?

def is_sorted(lst):
    return lst == sorted(lst) or lst == sorted(lst, reverse=True)

lst = [1, 2, 3, 4]
print(is_sorted(lst))

# Q17 Write a code to merge two sorted lists into a single sorted list?

def merge_sorted_lists(lst1, lst2):
    merged_list = []
    i = j = 0
    while i < len(lst1) and j < len(lst2):
        if lst1[i] < lst2[j]:
            merged_list.append(lst1[i])
            i += 1
        else:
            merged_list.append(lst2[j])
            j += 1
    merged_list.extend(lst1[i:])
    merged_list.extend(lst2[j:])
    return merged_list

lst1 = [1, 3, 5]
lst2 = [2, 4, 6]
print(merge_sorted_lists(lst1, lst2))

# Q18 implement a code to find the intersection of two given lists?

def intersect_lists(lst1, lst2):
    return list(set(lst1) & set(lst2))

lst1 = [1, 2, 3]
lst2 = [3, 4, 5]
print(intersect_lists(lst1, lst2))

# Q19 reate a code to find the union of two lists without duplicates?

def union_lists(lst1, lst2):
    return list(set(lst1) | set(lst2))

lst1 = [1, 2, 3]
lst2 = [3, 4, 5]
print(union_lists(lst1, lst2))

# Q20 write a code to shuffle a given list randomly without using any built-in shuffle functions?

import random

def shuffle_list(lst):
    lst_copy = lst[:]
    random.shuffle(lst_copy)
    return lst_copy

lst = [1, 2, 3, 4]
print(shuffle_list(lst))

# Q21 write a code that takes two tuples as input and returns a new tuple containing elements that arecommon to both input tuples?

def common_elements(t1, t2):
    return tuple(set(t1) & set(t2))

t1 = (1, 2, 3)
t2 = (3, 4, 5)
print(common_elements(t1, t2))

# Q22 ceate a code that prompts the user to enter two sets of integers separated by commas. Then, print theintersection of these two sets?

def print_intersection():
    set1 = set(map(int, input("Enter first set of integers separated by commas: ").split(',')))
    set2 = set(map(int, input("Enter second set of integers separated by commas: ").split(',')))
    print(set1 & set2)
print_intersection()


# Q23 Write a code to concatenate two tuples. The function should take two tuples as input and return a newtuple containing elements from both input tuples

def concatenate_tuples(t1, t2):
    return t1 + t2

t1 = (1, 2)
t2 = (3, 4)
print(concatenate_tuples(t1, t2))

# Q24  Develop a code that prompts the user to input two sets of strings. Then, print the elements that arepresent in the first set but not in the second set

def main():
    # Prompt the user for the first set of strings
    set1_input = input("Enter the first set of strings separated by commas: ")
    # Convert the input string into a set of strings
    set1 = set(map(str.strip, set1_input.split(',')))
    
    # Prompt the user for the second set of strings
    set2_input = input("Enter the second set of strings separated by commas: ")
    # Convert the input string into a set of strings
    set2 = set(map(str.strip, set2_input.split(',')))
    
    # Calculate the difference between the two sets
    difference = set1 - set2
    
    # Print the result
    print("Elements in the first set but not in the second set:", difference)

if __name__ == "__main__":
    main()

# Q25 Create a code that takes a tuple and two integers as input. The function should return a new tumaining elements from the original tuple within the specified range of indices

def tuple_within_range(t, start, end):
    return t[start:end]

t = (10, 20, 30, 40, 50)
start = 1
end = 4
print(tuple_within_range(t, start, end))

# Q27 Develop a code that takes a tuple of integers as input. The function should return the maximum andminimum values from the tuple using tuple unpacking

def union_of_sets():
    # Prompt the user for the first set of characters
    set1_input = input("Enter the first set of characters separated by commas: ")
    set1 = set(map(str.strip, set1_input.split(',')))
    
    # Prompt the user for the second set of characters
    set2_input = input("Enter the second set of characters separated by commas: ")
    set2 = set(map(str.strip, set2_input.split(',')))
    
    # Calculate the union of the two sets
    union = set1 | set2
    
    # Print the result
    print("Union of the two sets:", union)

if __name__ == "__main__":
    union_of_sets()

# Q28 create a code that defines two sets of integers. Then, print the union, intersection, and difference of thesetwo sets.

def set_operations():
    # Define two sets of integers
    set1 = {1, 2, 3, 4, 5}
    set2 = {4, 5, 6, 7, 8}
    
    # Calculate the union, intersection, and difference
    union = set1 | set2
    intersection = set1 & set2
    difference = set1 - set2
    
    # Print the results
    print("Set 1:", set1)
    print("Set 2:", set2)
    print("Union of Set 1 and Set 2:", union)
    print("Intersection of Set 1 and Set 2:", intersection)
    print("Difference (Set 1 - Set 2):", difference)

# Run the function
if __name__ == "__main__":
    set_operations()

# Q29 Write a code that takes a tuple and an element as input. The function should return the count ofoccurrences of the given element in the tuple
def count_occurrences(t, element):
    # Return the count of occurrences of 'element' in the tuple 't'
    return t.count(element)

if __name__ == "__main__":
    # Input from the user
    tuple_input = input("Enter the tuple elements separated by commas (e.g., 1,2,3,4,2): ")
    element_input = input("Enter the element to count: ")

    # Convert the input string to a tuple of integers
    tuple_elements = tuple(map(int, tuple_input.split(',')))
    
    # Convert the input string to the element to search (it could be a string or integer)
    element = int(element_input)

    # Call the function and print the result
    count = count_occurrences(tuple_elements, element)
    print(f"The element {element} occurs {count} times in the tuple.")

# Q30 evelop a code that prompts the user to input two sets of strings. Then, print the symmetric difference ofthese two sets

def symmetric_difference_of_sets():
    # Prompt the user for the first set of strings
    set1_input = input("Enter the first set of strings separated by commas: ")
    set1 = set(map(str.strip, set1_input.split(',')))
    
    # Prompt the user for the second set of strings
    set2_input = input("Enter the second set of strings separated by commas: ")
    set2 = set(map(str.strip, set2_input.split(',')))
    
    # Calculate the symmetric difference of the two sets
    symmetric_difference = set1 ^ set2
    
    # Print the result
    print("Symmetric difference of the two sets:", symmetric_difference)

if __name__ == "__main__":
    symmetric_difference_of_sets()

# Q31 Write a code that takes a list of words as input and returns a dictionary where the keys are unique wordsand the values are the frequencies of those words in the input list.

def word_frequencies(words_list):
    # Create an empty dictionary to store word frequencies
    frequency_dict = {}
    
    # Iterate over each word in the list
    for word in words_list:
        # Update the count of the word in the dictionary
        if word in frequency_dict:
            frequency_dict[word] += 1
        else:
            frequency_dict[word] = 1
    
    return frequency_dict

if __name__ == "__main__":
    # Input from the user
    input_string = input("Enter a list of words separated by spaces: ")
    words_list = input_string.split()
    
    # Get the word frequencies
    frequencies = word_frequencies(words_list)
    
    # Print the result
    print("Word frequencies:", frequencies)
# Q32 Write a code that takes two dictionaries as input and merghem into a single dictionary. If there arecommon keys, the values should be added together

def merge_dictionaries(dict1, dict2):
    # Create a new dictionary to store the merged result
    merged_dict = {}

    # Add all key-value pairs from the first dictionary
    for key, value in dict1.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value

    # Add all key-value pairs from the second dictionary
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value

    return merged_dict

if __name__ == "__main__":
    # Input from the user
    dict1_input = input("Enter the first dictionary (e.g., {'a': 1, 'b': 2}): ")
    dict2_input = input("Enter the second dictionary (e.g., {'b': 3, 'c': 4}): ")
    
    # Convert the input strings to dictionaries
    import ast
    dict1 = ast.literal_eval(dict1_input)
    dict2 = ast.literal_eval(dict2_input)
    
    # Merge the dictionaries
    merged = merge_dictionaries(dict1, dict2)
    
    # Print the result
    print("Merged dictionary:", merged)
# Q33 Write a code to access a value in a nested dictionary. The function should take the dictionary and a list ofkeys as input, and return the corresponding value. If any of the keys do not exist in the dictionary, thefunction should return None&

 def get_nested_value(d, keys):                                                                                                                                                                                                                                    
  
    current = d
    for key in keys:
        # Check if the current key is in the dictionary
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            # Return None if the key is not found or if the current value is not a dictionary
            return None
    return current

if __name__ == "__main__":
    # Input from the user
    import ast
    dict_input = input("Enter the nested dictionary (e.g., {'a': {'b': {'c': 10}}}): ")
    keys_input = input("Enter the list of keys separated by commas (e.g., a,b,c): ")
    
    # Convert the input strings to appropriate data types
    nested_dict = ast.literal_eval(dict_input)
    keys_list = [key.strip() for key in keys_input.split(',')]
    
    # Get the value from the nested dictionary
    result = get_nested_value(nested_dict, keys_list)
    
    # Print the result
    print("The value corresponding to the keys is:", result)

# Q34  Write a code that takes a dictionary as input and returns a sorted version of it based on the values. Youcan choose whether to sort in ascending or descending order
  def sort_dict_by_values(d, descending=False):
 
    # Sort the dictionary by values
    sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=descending))
    return sorted_dict

if __name__ == "__main__":
    import ast
    # Input from the user
    dict_input = input("Enter the dictionary (e.g., {'apple': 2, 'banana': 1, 'cherry': 3}): ")
    order_input = input("Enter the sorting order (asc for ascending, desc for descending): ").strip().lower()
    
    # Convert the input string to a dictionary
    dictionary = ast.literal_eval(dict_input)
    
    # Determine the sorting order
    descending = order_input == 'desc'
    
    # Get the sorted dictionary
    sorted_dict = sort_dict_by_values(dictionary, descending)
    
    # Print the result
    print("Sorted dictionary:", sorted_dict)

    
