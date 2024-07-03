#Užduotis 5
	#Funkcija: Sukurkite funkciją get_value_dict(dict, key), remove_value_dict(dict, key), kuri iš žodyno išgauna reikšmę arba pašalina.
	#Fixture: Sukurkite fixture dict_data(), kuriame pateikiamas žodynas ir raktas.
	#Testas: Parašykite testavimo funkcijas, kuri naudotų dict_data testavime.

def get_value_dict(dict_obj, key):
    return dict_obj.get(key)

def remove_value_dict(dict_obj, key):
    if key in dict_obj:
        del dict_obj[key]
    return dict_obj

import pytest

@pytest.fixture
def dict_data():
    return {
        "vardas": "Darius",
        "amzius": 32,
        "miestas": "Vilnius"
    }, "amzius"

def test_get_value_dict(dict_data):
    dict_obj, key = dict_data
    assert get_value_dict(dict_obj, key) == 32
    assert get_value_dict(dict_obj, "non_existing_key") is None

def test_remove_value_dict(dict_data):
    dict_obj, key = dict_data
    modified_dict = remove_value_dict(dict_obj, key)
    assert key not in modified_dict
    assert modified_dict == {
        "vardas": "Darius",
        "miestas": "Vilnius"
    }