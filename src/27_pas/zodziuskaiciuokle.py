#Užduotis 3	Funkcija: Sukurkite funkciją count_words_in_file(file_path), kuri skaičiuoja žodžių skaičių duotame tekstiniame faile.
	#Fixture: Sukurkite fixture temp_file_with_content(), kuris sukuria laikiną teksto failą su iš anksto nustatytu turiniu ir grąžina failo kelią.
	#Testas: Parašykite testo funkciją, kuri naudotų temp_file_with_content fixture, kad patikrintų finkciją count_words_in_file.

def count_words_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    words = content.split()
    return len(words)

import pytest
import tempfile
import os

@pytest.fixture
def temp_file_with_content():
    content = "Testinis failas."
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
    temp_file.write(content)
    temp_file.close()
    yield temp_file.name
    os.remove(temp_file.name)

def test_count_words_in_file(temp_file_with_content):
    file_path = temp_file_with_content
    word_count = count_words_in_file(file_path)
    assert word_count == 2