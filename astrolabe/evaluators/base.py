from __future__ import annotations

from abc import abstractmethod


class Evaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, inputs=None, runtime_config=None):
        input_feat = self.prepare_input(inputs, runtime_config)
        output = self.run_function(input_feat)
        std_output = self.prepare_output(output, inputs, runtime_config)
        return std_output

    @abstractmethod
    def run_function(self, input_feat):
        pass

    @abstractmethod
    def prepare_input(self, inputs, runtime_config): ...

    @abstractmethod
    def prepare_output(self, output, inputs, runtime_config): ...
