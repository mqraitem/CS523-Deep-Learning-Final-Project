from pathlib import Path

from .utilities.data_handler import DataHandler
from .ai_model.language_model import LanguageModel
from .ai_model.vision_model import VisionModel
from .ai_model.fbhm_ham import FbhmHAM 

def main():
    # data_path = Path('./data')
    # dh = DataHandler(data_path)
    # d, tr,  t = dh.load_all_data()
    # # tr, v, t = dh.load_given_data()
    # dh.compute_data_analytics(d)
    # print(dh.da)
    lm = LanguageModel()
    vm = VisionModel()
    ham = FbhmHAM()
    return 0

if __name__ == "__main__":
    main()
