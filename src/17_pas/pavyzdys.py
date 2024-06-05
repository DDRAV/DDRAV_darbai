from abc import ABC, abstractmethod


class Vehicle(ABC):
    @abstractmethod
    def start_engine(self, a, b):
        pass

    @abstractmethod
    def stop_engine(self):
        pass

    @staticmethod
    def vehicle_info():
        return "This is a vehicle"


class Car(Vehicle):
    def start_engine(self, a, b):
        return "Car engine started"

    def stop_engine(self):
        return "Car engine stopped"


class Bike(Vehicle):
    def start_engine(self, a, b):
        return "Car engine started"

    def stop_engine(self):
        return "Car engine stopped"


bike = Bike()
bike.start_engine('a', 'b')