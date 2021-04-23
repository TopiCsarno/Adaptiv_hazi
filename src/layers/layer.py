"""
Absztrakt osztály, neurális háló rétegeihez. 
"""
from abc import ABC, abstractmethod

class Layer(ABC):

    @property
    def weights(self):
        return None
    
    @abstractmethod
    def forward_pass(self, a_prev):
        pass

    @abstractmethod
    def back_pass(self, da_curr):
        pass