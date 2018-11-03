import os
import json
import abc


class BaseModel(abc.ABC):
    '''
    Rafiki's base concept drift detection method class that Rafiki detection methods should extend. 
    Rafiki detection methods should implement all abstract methods.
    '''   

    @abc.abstractmethod
    def update_on_query(self):
        raise NotImplementedError


    @abc.abstractmethod
    def update_on_feedback(self):
        raise(NotImplementedError)


    @abc.abstractmethod
    def destroy(self):
        '''
        Destroy this model instance, closing any sessions or freeing any connections.
        No other methods will be called subsequently.
        '''
        pass





