import os
from typing import Optional, NoReturn

import attr
import ujson as ujson

import vowpalwabbit as vw


@attr.s(auto_attribs=True)
class Model:
    load_model_path: Optional[str] = None
    min_value: float = 0
    max_value: float = 32
    bandwidth: float = 2
    num_actions: int = 128
    epsilon: float = 0.1
    random_seed: int = 648
    flags_list: list = ['--chain_hash', '--coin']
    quiet: bool = False
    model: vw.Workspace = None

    def __attrs_post_init__(self):
        init_str = (
            f"--cats {self.num_actions} --bandwidth {self.bandwidth} "
            f"--min_value {self.min_value} --max_value {self.max_value} "
            f"--json {' '.join(self.flags_list)} --epsilon {self.epsilon} "
            f"-q :: "
        )
        if self.quiet:
            init_str += "--quiet "
        # if self.load_model_path:
        #     init_str += f"-f {self.load_model_path} "
        init_model_name = '1' + self.load_model_path
        if init_model_name and os.path.exists(init_model_name):
            init_str += f"-i {init_model_name} "

        self.model = vw.Workspace(
            init_str,
            enable_logging=True,
        )

    def predict(self, context: dict) -> dict:
        example = self._make_json(context)
        vw_string = self.model.parse(example, vw.LabelType.CONTINUOUS)
        chosen_temp, pdf_value = self.model.predict(
            vw_string,
            vw.PredictionType.ACTION_PDF_VALUE,
        )
        return {
            'action': chosen_temp,
            'pdf_value': pdf_value,
        }

    def update_model(self, data: dict) -> NoReturn:
        context = data["context"]
        prediction = data["label"]
        example = self._make_json(context, prediction)
        vw_string = self.model.parse(example, vw.LabelType.CONTINUOUS)
        self.model.learn(vw_string)
        self.model.finish_example(vw_string)

    def _make_json(self, context: dict, label: Optional[dict] = None) -> str:
        example_dict = {}
        if label is not None:
            example_dict["_label_ca"] = label
        example_dict["c"] = {
            "room=": context["room"],
            "time_of_day=": context["time_of_day"],
        }
        return ujson.dumps(example_dict)
