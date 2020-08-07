from pathlib import Path
import pandas as pd


class Psept:
    def __init__(self):
        self.__data = None

    def load_savs(self, filepath):
        try:
            with open(Path(filepath)) as f:
                data = f.read().splitlines()
        except:
            raise TypeError("File loaded error. Please check file path or file format. Only txt or csv file are allowed.")
        self.__data = data

    def show_data(self, datatype="list"):
        if datatype == "list":
            print(self.__data)
        elif datatype in ["pd", "pandas"]:
            df = pd.DataFrame()
            uniprot, position, wt_aa, mu_aa = zip(*[sav.split() for sav in self.__data])
            df["uniprot_id"], df["mutated_position"], df["wildtype_aa"], df["mutated_aa"] = uniprot, position, wt_aa, mu_aa
            print(df)
        else:
            raise TypeError("Wrong data type.")
            
                
            

def main():
    psept = Psept()
    psept.load_savs("test_sav.txt")
    psept.show_data("pd")


if __name__ == "__main__":
    main()
