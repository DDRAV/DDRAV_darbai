#1
def is_palindrome(s):
    cleaned = ''.join(c for c in s if c.isalnum()).lower()
    return cleaned == cleaned[::-1]

def test_palindrome():
    assert is_palindrome("A man a plan a canal Panama") == True
    assert is_palindrome("racecar") == True
    assert is_palindrome("madam") == True

def test_non_palindrome():
    assert is_palindrome("hello") == False
    assert is_palindrome("world") == False

def test_empty_string():
    assert is_palindrome("") == True

#2
def max_of_three(a, b, c):
    return max(a, b, c)

def test_positive_numbers():
    assert max_of_three(1, 2, 3) == 3
    assert max_of_three(10, 20, 30) == 30
    assert max_of_three(5, 15, 10) == 15

def test_negative_numbers():
    assert max_of_three(-1, -2, -3) == -1
    assert max_of_three(-10, -20, -30) == -10
    assert max_of_three(-5, -15, -10) == -5

def test_mixed_numbers():
    assert max_of_three(-1, 0, 1) == 1
    assert max_of_three(-10, 10, 0) == 10
    assert max_of_three(5, -15, 10) == 10
    assert max_of_three(-5, 5, -5) == 5

#7

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def test_bendro_nera():
    assert gcd(8, 9) == 1
    assert gcd(14, 15) == 1

def test_vienas_nulis():
    assert gcd(0, 5) == 5
    assert gcd(7, 0) == 7

def test_identiski():
    assert gcd(10, 10) == 10
    assert gcd(100, 100) == 100

def test_bendras():
    assert gcd(48, 18) == 6
    assert gcd(54, 24) == 6


#8

def merge_sorted_lists(list1, list2):
    merged_list = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1
    merged_list.extend(list1[i:])
    merged_list.extend(list2[j:])
    return merged_list

def test_both_non_empty():
    assert merge_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted_lists([1, 2, 3], [4, 5, 6]) == [1, 2, 3, 4, 5, 6]

def test_one_empty():
    assert merge_sorted_lists([], [1, 2, 3]) == [1, 2, 3]
    assert merge_sorted_lists([4, 5, 6], []) == [4, 5, 6]

def test_both_empty():
    assert merge_sorted_lists([], []) == []

def test_general_cases():
    assert merge_sorted_lists([1, 4, 5], [2, 3, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted_lists([1, 2, 6], [3, 4, 5]) == [1, 2, 3, 4, 5, 6]

#9
def count_vowels(s):
    vowels = "aeiouyAEIOUY"
    return sum(1 for char in s if char in vowels)

def test_with_vowels():
    assert count_vowels("hello") == 2
    assert count_vowels("aeiou") == 5
    assert count_vowels("AEIOU") == 5

def test_without_vowels():
    assert count_vowels("bcdfg") == 0
    assert count_vowels("rhthm") == 0

def test_empty_string():
    assert count_vowels("") == 0

def test_mixed_case():
    assert count_vowels("HeLLo WoRLd") == 3
    assert count_vowels("PyThOn PrOgRaMmInG") == 5

#4
def fibonacci(n):
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def test_small_values():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5


def test_large_values():
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765
    assert fibonacci(30) == 832040


def test_edge_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1

    with pytest.raises(ValueError):
        fibonacci(-1)


def test_performance():
    assert fibonacci(50) == 12586269025
    assert fibonacci(100) == 354224848179261915075