# python -m venv myenv
# pip install <name_of_the_package>

from dataclasses import dataclass, field


@dataclass
class Person:
    name: str
    age: int
    address: str = field(default="Unknown Address")


person1 = Person(name="Alice", age=30)
print(person1)