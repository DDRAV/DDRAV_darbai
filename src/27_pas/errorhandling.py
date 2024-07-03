#Užduotis 1: Parašykite funkciją create_temp_file, kuri sukuria laikiną failą duotame kataloge. Naudokite fixtures, kad testams pateiktumėte laikiną katalogą, filų pavadinimus ir turinius.
	#Testai: Užtikrinkite, kad failai būtų sukurti ir turinys teisingas.
import pytest
import os

def create_file(filename: str, dir_name: str, content: str):
    filepath = f'{dir_name}/{filename}'
    with open (filepath, 'w') as f:
        f.write(content)


@pytest.fixture
def file_data():
    dir_name = 'test_dir'
    os.mkdir(dir_name)
    filename = 'test_file.txt'
    content = 'This is file'
    return filename, dir_name, content

def test_create_file(file_data):
    filename, dir_name, content_expecter = file_data
    create_file(filename, dir_name, content_expecter)

    assert os.path.exists(f'{dir_name}/{filename}')
    with open(filename, 'r') as f:
        content = f.read()

    assert content == content_expecter
