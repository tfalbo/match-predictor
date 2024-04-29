from unittest import TestCase

from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.result import Result
from matchpredictor.matchresults.results_provider import training_results, validation_results
from matchpredictor.predictors.random_forest_predictor import train_random_forest_predictor
from test.predictors import csv_location


class TestRandomForestPredictor(TestCase):
    def test_accuracy(self) -> None:
        training_data = training_results(csv_location, 2019, result_filter=lambda result: result.season >= 2015)
        validation_data = validation_results(csv_location, 2019)
        predictor = train_random_forest_predictor(training_data)

        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, .33)
