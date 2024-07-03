#Užduotis 2: Parašykite fixture math_params(), kuri grąžina du sveikuosius skaičius a ir b, kurie gali būti naudojami matematinių operacijų testuose.
	#Funkcijos: Implementuokite funkcijas add(a, b), sub(a, b), multiply(a, b), devide(a, b).
	#Testas: Parašykite testus, kuriuose naudokite fixture ir patikrinkite įgyvendintas funkcijas.

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Negalima dalint is nulio")
    return a / b

import pytest

@pytest.fixture
def math_params():
    return 6, 3

def test_add(math_params):
    a, b = math_params
    assert add(a, b) == 9

def test_sub(math_params):
    a, b = math_params
    assert sub(a, b) == 3

def test_multiply(math_params):
    a, b = math_params
    assert multiply(a, b) == 18

def test_divide(math_params):
    a, b = math_params
    assert divide(a, b) == 2

def test_divide_by_zero():
    with pytest.raises(ValueError, match="Negalima dalint is nulio"):
        divide(6, 0)
        
