#Užduotis 4
	#Funkcija: Įgyvendinkite funkcijas remove_from_list(lst), append_to_list(lst), remove_duplicates_from_list(lst) kuri pašalina pasikartojančius elementus iš sąrašo lst, prideda elementus arba pašalina.
	#Fixture: Sukurkite fixture sample_list(), kurioje pateikiams sąrašas (arba keli sąrašai).
	#Testas: Parašykite testavimo funkcijas, kuri naudotų sample_list, kad patikrintų remove_from_list(lst), append_to_list(lst), remove_duplicates_from_list(lst).

def remove_from_list(lst, element):
    if element in lst:
        lst.remove(element)
    return lst

def append_to_list(lst, element):
    lst.append(element)
    return lst

def remove_duplicates_from_list(lst):
    return list(set(lst))

import pytest

@pytest.fixture
def sample_list():
    return [1, 2, 3, 2, 4, 5, 3, 6, 7, 8, 5]

def test_remove_from_list(sample_list):
    lst = sample_list.copy()
    result = remove_from_list(lst, 2)
    assert result == [1, 3, 2, 4, 5, 3, 6, 7, 8, 5]

def test_append_to_list(sample_list):
    lst = sample_list.copy()
    result = append_to_list(lst, 9)
    assert result == [1, 2, 3, 2, 4, 5, 3, 6, 7, 8, 5, 9]

def test_remove_duplicates_from_list(sample_list):
    result = remove_duplicates_from_list(sample_list)
    assert set(result) == {1, 2, 3, 4, 5, 6, 7, 8} and len(result) == 8