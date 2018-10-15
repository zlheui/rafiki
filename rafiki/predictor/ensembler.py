import abc
import numpy as np
from rafiki.constants import TaskType, OutputType
from rafiki.config import TASK_TYPE_TO_OUTPUT_TYPE_MAPPING

def standardise_predictions(predictions_list, predict_label_mappings):
    main_predictions = predictions_list[0]
    main_predict_label_mapping = predict_label_mappings[0]
    main_label_to_index_mapping = {v: k for k, v in main_predict_label_mapping}

    new_predictions_list = np.empty([len(predictions_list), len(main_predictions)])
    new_predictions_list[0]= main_predictions

    for i in range(1, len(predictions_list)):
        predictions = predictions_list[i]
        predict_label_mappings = predict_label_mappings[i]
        new_predictions = np.empty([len(predictions),])

        for j in range(0, predictions):
            new_index = main_label_to_index_mapping[predict_label_mappings[j]]
            new_predictions[new_index] = predictions[j]
        
        new_predictions_list[i] = new_predictions

    return (new_predictions_list, main_predict_label_mapping)

class BaseEnsembler(abc.ABC):

    @abc.abstractmethod
    def init(self, task):
        pass

    @abc.abstractmethod
    def ensemble_predictions(predictions_list, predict_label_mappings):
        raise NotImplementedError()

class ArithmeticMeanEnsembler(BaseEnsembler):

    def init(self, task):
        self._task = task
        self._output_type = TASK_TYPE_TO_OUTPUT_TYPE_MAPPING[task]
    
    def ensemble_predictions(predictions_list, predict_label_mappings):
        (predictions_list, predict_label_mapping) = standardise_predictions(predictions_list, predict_label_mappings)
        predictions = predictions_list[0]

        if self._output_type == OutputType.PROBABILITY_VECTOR:
            predictions = np.mean(predictions_list, axis=0)
        
        return predictions

class MajorityVotingEnsembler(BaseEnsembler):

    def init(self, task):
        self._task = task
        self._output_type = TASK_TYPE_TO_OUTPUT_TYPE_MAPPING[task]
    
    def ensemble_predictions(predictions_list, predict_label_mappings):
        (predictions_list, predict_label_mapping) = standardise_predictions(predictions_list, predict_label_mappings)
        predictions = predictions_list[0]

        if self._output_type == OutputType.PROBABILITY_VECTOR:
            
            counter = np.zeros([len(predictions_list[0])], dtype=np.int)
            for current_predictions in predictions_list:
                pred_indices = np.argmax(current_predictions, axis=1)
                counter[pred_indices] += 1
            
            return np.amax(counter)

