from pathlib import Path
import pandas as pd
import json
import re
import os
from urllib.request import urlretrieve
from urllib.error import HTTPError
from pyrosetta import init
from pyrosetta.io import pose_from_pdb
from pyrosetta.toolbox.cleaning import cleanATOM


class Psept:
    __repack_radius = 8

    def __init__(self, pdb_info_path='pdb_info.json'):
        """
        Initilize and setting for Psept.
        """
        self.__data = None

        try:
            with open(Path(pdb_info_path), 'r') as f:
                self.__pdb_info = json.load(f)
            self
        except FileNotFoundError:
            self.__pdb_info = {
                'size': {'source': '', 'date': '', 'data': ''},
                'download_failed': {'source': '', 'date': '', 'data': ''},
                'load_failed': {'source': '', 'date': '', 'data': ''},
                'multimodel': {'source': '', 'date': '', 'data': ''}
            }
        self.__pdb_root_folder = Path('./pdb_files')
        self.__skip_pdb_list = self.__pdb_info['download_failed']['data'] + self.__pdb_info['load_failed']['data'] + self.__pdb_info['multimodel']['data']

    def load_savs(self, filepath):
        """
        Load input sav file. 
        Input format: {UniProt ID} {mutated position} {wildtype amino acid} {mutated amino acid}
        e.g., P04637 35 L F
        """
        try:
            with open(Path(filepath)) as f:
                data = f.read().splitlines()
        except:
            raise TypeError('File loaded error. Please check file path or file format. Only txt or csv file are allowed.')
        self.__data = data

    def show_data(self, datatype='list'):
        """
        Show loaded input data.
        Defalut type is list. DataFrame is available.
        """
        if not self.__data:
            print('No data were loaded.')
        elif datatype == 'list':
            print(self.__data)
        elif datatype in ['pd', 'pandas']:
            df = pd.DataFrame()
            uniprot, position, wt_aa, mu_aa = zip(*[sav.split() for sav in self.__data])
            df['uniprot_id'], df['mutated_position'], df['wildtype_aa'], df['mutated_aa'] = uniprot, position, wt_aa, mu_aa
            print(df)
        else:
            raise TypeError('Wrong data type.')

    def get_pdb_info(self):
        """
        Return loaded pdb_info
        """
        return self.__pdb_info

    def set_pdb_folder(self, folder_path):
        self.__pdb_root_folder = Path(folder_path)

    @classmethod
    def download_pdb(cls, pdb):
        """
        Download PDB file from RCSB.
        Input: PDB code.
        """
        root_folder = Path('pdb_files')
        pdb_folder = root_folder / 'full_pdb'
        clean_folder = root_folder / 'clean_pdb'
        if not os.path.isdir(root_folder):
            os.makedirs(root_folder)
            os.makedirs(pdb_folder)
            os.makedirs(clean_folder)
        filepath = pdb_folder / (pdb + '.pdb')
        if not os.path.isfile(filepath):
            url = 'https://files.rcsb.org/download/' + (pdb + '.pdb')
            urlretrieve(url, filepath)

    @classmethod
    def check_pdb_is_multimodel(cls, pdb_path):
        """
        Check input PDB is multimodel or not.
        """
        with open(pdb_path) as f:
            if "NUMMDL" in f.read():
                return True
            else:
                return False

    @classmethod
    def get_repack_radius(cls):
        """
        Get repack_radius
        """
        return cls.__repack_radius

    @classmethod
    def set_repack_radius(cls, new_repack_radius):
        """
        Set repack_radius
        """
        cls.__repack_radius = new_repack_radius

    def get_pdb_size(self, pdb):
        """
        Get pdb size. This function follows several steps:
            1. Find exist clean PDB file
                if not exist, download PDB from RCSB
                    if failed, return None
            2. Check PDB is multimodel or not
                if yes, return None
            3. Generate clean atom PDB file
            4. Load PDB using Rosetta
                if failed, return None
            5. Return size of PDB
        All failed will be recorded in pdb_info.json and remove corresponding PDB file (either full or clean).
        """
        full_pdb_path = self.__pdb_root_folder / 'full_pdb' / (pdb + '.pdb')
        clean_pdb_path = self.__pdb_root_folder / 'clean_pdb' / (pdb + '.clean.pdb')
        if not os.path.isfile(clean_pdb_path):
            try:
                self.download_pdb(pdb)
            except HTTPError:
                self.__pdb_info['download_failed']['data'].append(pdb)
                os.remove(full_pdb_path)
                return None
            is_multimodle = self.check_pdb_is_multimodel(full_pdb_path)
            if is_multimodle:
                self.__pdb_info['multimodel']['data'].append(pdb)
                os.remove(full_pdb_path)
                return None
            cleanATOM(str(full_pdb_path), str(clean_pdb_path))
        try:
            pose = pose_from_pdb(str(clean_pdb_path))
        except RuntimeError:
            self.__pdb_info['load_failed']['data'].append(pdb)
            os.remove(full_pdb_path)
            os.remove(clean_pdb_path)
            return None
        size = pose.total_residue()
        self.__pdb_info['size']['data'][pdb] = size
        return size

    def create_ordered_pdb_list(self):
        """
        Create ordered pdb list
        """
        for sav in self.__data:
            uniprot, pos, wt_aa, mu_aa = sav.split()
            pos = int(pos)
            pdb_size_list = []

            # Get pdb list for sav gene and process each pdb.
            pdb_list = self.__pdb_info['uniprot_pdb']['data'][uniprot]
            for pdb in pdb_list:

                # Continue if no pdb were found.
                if pdb in self.__skip_pdb_list:
                    continue

                # Find corresponding dataframe.
                temp_df = self.__map_df.loc[(self.__map_df['PDB'] == pdb) & (self.__map_df['SP_PRIMARY'] == uniprot)]
                # Filter data which wasn't containing mutated position.
                temp_df = temp_df.loc[(self.__map_df['SP_BEG'] < pos) & (self.__map_df['SP_END'] > pos)]
                if temp_df.empty:
                    continue

                pdb_series = temp_df.iloc[0]
                chain = pdb_series['CHAIN']
                pdb_beg = int(re.sub('[^0-9]', '', pdb_series['PDB_BEG']))
                pdb_end = int(re.sub('[^0-9]', '', pdb_series['PDB_END']))
                uni_beg = int(pdb_series['SP_BEG'])
                uni_end = int(pdb_series['SP_END'])

                # (pdb_end - pdb_beg) != (uni_end - uni_beg) indicate that wrong positions were recorded.
                if (pdb_end - pdb_beg) != (uni_end - uni_beg):
                    continue

                pos_in_pdb = pos + (pdb_beg - uni_beg)

                # Get PDB size from pdb_info
                try:
                    pdb_size = int(self.__pdb_info['size']['data'][pdb])

                # If not recorded, download PDB.
                except KeyError:
                    pdb_size = self.get_pdb_size(pdb)

                # Continue if any error occurred.
                if not pdb_size:
                    continue
                else:
                    pdb_size_list.append((pdb, pdb_size, chain, pos_in_pdb))

            # Continue if no PDB matched.
            if not pdb_size_list:
                self.__sav_to_ordered_pdb.append([sav, 'No PDB matched.'])
                continue

            # Sort by size and order by median.
            sort_by_size = sorted(pdb_size_list, key=lambda x: x[1])
            middle_start_list = []
            while sort_by_size:
                median = len(sort_by_size)//2
                middle_start_list.append(sort_by_size.pop(median))
            self.__sav_to_ordered_pdb.append([sav, middle_start_list])
            print(self.__sav_to_ordered_pdb)
            break

    def scoring(self):
        """
        Main function for data preprocessing and Rosetta scoring.
        The function will process savs in input file and output energy scores.
        """
        init()
        self.__sav_to_ordered_pdb = []
        # Load position data.
        self.__map_df = pd.read_csv('uniprot_segments_observed.csv', skiprows=1)

        # Generate ordered pdb list for input savs.
        self.create_ordered_pdb_list()


def main():
    psept = Psept()
    psept.load_savs('test_sav.txt')
    psept.scoring()


if __name__ == '__main__':
    main()
