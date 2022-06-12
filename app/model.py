from typing import Optional, NoReturn

import attr
import ujson as ujson

import vowpalwabbit as vw


@attr.s(auto_attribs=True)
class Model:
    min_value: float = 0
    max_value: float = 32
    bandwidth: float = 2
    num_actions: int = 128
    epsilon: float = 0.1
    flags_list: list = ['--chain_hash', '--coin']
    quiet: bool = False
    model: vw.Workspace = None

    def __attrs_post_init__(self):
        self.model = vw.Workspace(
            f"--cats {self.num_actions} --bandwidth {self.bandwidth} "
            f"--min_value {self.min_value} --max_value {self.max_value} "
            f"--json {' '.join(self.flags_list)} --epsilon {self.epsilon} "
            "-q ::" + (" --quiet" if self.quiet else "")
        )

    def predict(self, context: dict) -> dict:
        example = self._make_json(context)
        chosen_temp, pdf_value = self.model.predict(example)
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
