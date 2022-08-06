import os
import pandas as pd
from typing import Union, List

class YodaDataset:
    def __init__(self, paths: Union[List[str], str]) -> None:
        self._ds = pd.read_csv(paths)
    def schema(self):
        return None
    def num_blocks(self) -> int:
        return 1
    def to_pandas(self) -> pd.DataFrame:
        return self._ds

class YodaDatasets:
    @staticmethod
    def get(name: str, version: int = None, use_last_version: bool = True):
        # import ray
        # return ray.data.read_csv(paths='pkg/dataset/wine-quality.csv')
        return YodaDataset(paths=os.path.join(os.path.dirname(__file__), "wine-quality.csv"))
