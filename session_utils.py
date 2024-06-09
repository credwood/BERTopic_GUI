import json
import os
import pickle
from collections import OrderedDict
from typing import Tuple, List, Union
from dataclasses import dataclass

from numpy import ndarray

from data_utils import extract_text_from_document, get_document_paths


@dataclass
class Session:
    def __init__(self,
                 model_path: Union[str, None] = None, 
                 documents: OrderedDict = OrderedDict(),
                 is_saved: bool = False,
                 ):
        self.MODEL_PATH = model_path
        self.IS_SAVED = is_saved
        self.DOCUMENTS = documents
        self.FIT_TRANSFORM_RESULTS = None
        self.LOADED_MODEL = None
    
def save_session(session: Session):
    if not os.path.isdir("data"):
        os.makedirs("data")
    with open(f"data/session.pickle", "wb") as f:
        pickle.dump(session, f)

def load_session():
    with open(f"data/session.pickle", "rb") as f:
        try:
            session = pickle.load(f)
        except EOFError:
            print("session could not load, creating new session")
            session = Session()
    return session
