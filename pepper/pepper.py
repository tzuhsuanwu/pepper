from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import os
from urllib.request import urlretrieve
from urllib.error import HTTPError
from pyrosetta import init
from pyrosetta.io import pose_from_pdb
from pyrosetta.toolbox.cleaning import cleanATOM
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.rosetta.core.scoring import ScoreFunction
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.core.scoring import calc_total_sasa


class Psept:
    __repack_radius = 8
    __probe_radius = 2.2

    __wt_columns = [
        "wt_fa_atr", "wt_fa_rep", "wt_fa_intra_rep", "wt_fa_sol", "wt_lk_ball_wtd", "wt_fa_intra_sol_xover4",
        "wt_fa_elec", "wt_hbond_lr_bb", "wt_hbond_sr_bb", "wt_hbond_bb_sc", "wt_hbond_sc", "wt_dslf_fa13",
        "wt_rama_prepro", "wt_p_aa_pp", "wt_fa_dun", "wt_omega", "wt_pro_close", "wt_yhh_planarity", "wt_ref", "wt_sum"
    ]

    __mu_columns = [
        "mu_fa_atr", "mu_fa_rep", "mu_fa_intra_rep", "mu_fa_sol", "mu_lk_ball_wtd", "mu_fa_intra_sol_xover4",
        "mu_fa_elec", "mu_hbond_lr_bb", "mu_hbond_sr_bb", "mu_hbond_bb_sc", "mu_hbond_sc", "mu_dslf_fa13",
        "mu_rama_prepro", "mu_p_aa_pp", "mu_fa_dun", "mu_omega", "mu_pro_close", "mu_yhh_planarity", "mu_ref", "mu_sum"
    ]

    __diff_columns = [
        "diff_fa_atr", "diff_fa_rep", "diff_fa_intra_rep", "diff_fa_sol", "diff_lk_ball_wtd", "diff_fa_intra_sol_xover4",
        "diff_fa_elec", "diff_hbond_lr_bb", "diff_hbond_sr_bb", "diff_hbond_bb_sc", "diff_hbond_sc", "diff_dslf_fa13",
        "diff_rama_prepro", "diff_p_aa_pp", "diff_fa_dun", "diff_omega", "diff_pro_close", "diff_yhh_planarity", "diff_ref", "diff_sum"
    ]

    __whole_column = __wt_columns + __mu_columns + __diff_columns

    def __init__(self, pdb_info_path='pdb_info.json'):
        """
        Initilize and setting for Psept
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
        self.__root_pdb_folder = Path('./pdb_files')
        self.__full_pdb_folder = self.__root_pdb_folder / 'full_pdb'
        self.__clean_pdb_folder = self.__root_pdb_folder / 'clean_pdb'
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
        Show loaded input data
        Defalut type is list. DataFrame is available
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
        self.__root_pdb_folder = Path(folder_path)
        self.__full_pdb_folder = self.__root_pdb_folder / 'full_pdb'
        self.__clean_pdb_folder = self.__root_pdb_folder / 'clean_pdb'

    @classmethod
    def download_pdb(cls, pdb):
        """
        Download PDB file from RCSB
        Input: PDB code
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
        Check input PDB is multimodel or not
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

    @classmethod
    def get_probe_radius(cls):
        """
        Get probe_radius
        """
        return cls.__probe_radius

    @classmethod
    def set_probe_radius(cls, new_probe_radius):
        """
        Set probe_radius
        """
        cls.__probe_radius = new_probe_radius

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
        All failed will be recorded in pdb_info.json and remove corresponding PDB file (either full or clean)
        """
        full_pdb_path = self.__root_pdb_folder / 'full_pdb' / (pdb + '.pdb')
        clean_pdb_path = self.__root_pdb_folder / 'clean_pdb' / (pdb + '.clean.pdb')
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
            uniprot, pos = sav.split()[:2]
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

    def mutate(self):
        """
        Mutate each SAVs of input file
        """

        # Initialize and create result dataframe
        result_df_column = [
            'SAV', 'ordered_PDB_list', 'ordered_list_length', 'predict_status',
            'current_PDB', 'current_PDB_info', 'chain', 'position', 'PDB_size', 'chain_length',
        ]
        result_df_column = result_df_column + self.__whole_column + ['wt_SASA', 'mu_SASA', 'diff_SASA']
        result_df = pd.DataFrame(columns=result_df_column)
        result_df['SAV'] = [i[0] for i in self.__sav_to_ordered_pdb]
        result_df['ordered_PDB_list'] = [i[1] for i in self.__sav_to_ordered_pdb]

        while True:
            # Record "failed" if no more PDB in candidate list
            result_df.loc[
                (result_df['ordered_list_length'] == 0) & (result_df['predict_status'] == 'unfinished'),
                'predict_status'
            ] = 'failed'

            # Extract sub_df of unfinished SAVs
            sub_df = result_df[~result_df.predict_status.isin(['success', 'failed'])]

            if sub_df.empty:
                self.__scoring_result = result_df
                break

            # Pop new PDB from ordered_PDB_list and update the information
            sub_df['current_PDB_info'] = sub_df['ordered_PDB_list'].map(lambda x: x[0])
            sub_df['current_PDB'] = sub_df['current_PDB_info'].map(lambda x: x[0])
            sub_df['PDB_size'] = sub_df['current_PDB_info'].map(lambda x: x[1])
            sub_df['chain'] = sub_df['current_PDB_info'].map(lambda x: x[2])
            sub_df['position'] = sub_df['current_PDB_info'].map(lambda x: x[3])
            sub_df['ordered_PDB_list'] = sub_df['ordered_PDB_list'].apply(lambda x: x[1:])
            sub_df['ordered_list_length'] = sub_df['ordered_PDB_list'].apply(lambda x: len(x))

            for pdb in sub_df['current_PDB'].unique():

                # Calculate wildtype PDB score
                wt_pose = pose_from_pdb(str(self.__clean_pdb_folder / (pdb + '.clean.pdb')))
                wt_score = np.array(self.pyrosetta_scoring(wt_pose))
                wt_sum = np.sum(wt_score)
                wt_score = np.append(wt_score, wt_sum)

                temp_df = sub_df[sub_df['current_PDB'] == pdb]
                for idx, row in temp_df.iterrows():
                    chain = row['chain']
                    position_in_pdb = row['position']
                    wt_aa, mu_aa = row['SAV'].split()[2:4]

                    # Get actual mutate position in pose
                    position_in_pose = wt_pose.pdb_info().pdb2pose(chain, int(position_in_pdb))

                    # If position not found or residue in pose not equal to SAV, continue
                    if position_in_pose == 0 or wt_pose.residue(position_in_pose).name1() != wt_aa:
                        sub_df.set_value(idx, 'predict_status', 'unfinished')
                        continue

                    # Get chain number and chain length
                    chain_num = wt_pose.chain(position_in_pose)
                    chain_length = len(wt_pose.chain_sequence(chain_num))

                    # Mutate pose
                    mutant_pose = wt_pose.clone()
                    mutate_residue(mutant_pose, position_in_pose, mu_aa, self.__repack_radius)

                    # Calculate mutant PDB score
                    mu_score = np.array(self.pyrosetta_scoring(mutant_pose))
                    mu_sum = np.sum(mu_score)
                    mu_score = np.append(mu_score, mu_sum)

                    # Calculate difference score
                    diff_score = mu_score - wt_score

                    result = wt_score.tolist() + mu_score.tolist() + diff_score.tolist()

                    # Successful score, update the predict_status
                    sub_df.set_value(idx, 'predict_status', 'success')
                    sub_df.set_value(idx, 'chain_length', chain_length)

                    # Calculate score of SASA
                    wt_sasa = calc_total_sasa(wt_pose, self.__probe_radius)
                    mu_sasa = calc_total_sasa(mutant_pose, self.__probe_radius)
                    sub_df.set_value(idx, "wt_SASA", wt_sasa)
                    sub_df.set_value(idx, "mu_SASA", mu_sasa)
                    sub_df.set_value(idx, "diff_SASA", mu_sasa - wt_sasa)

                    for i in range(len(self.__whole_column)):
                        sub_df.set_value(idx, self.__whole_column[i], result[i])
                result_df.update(sub_df)

    def pyrosetta_scoring(self, pose):
        """
        Scoring a pose by REF2015
        """
        scorefxn = ScoreFunction()

        scorefxn.set_weight(ScoreType.fa_atr, 1.0)
        scorefxn.set_weight(ScoreType.fa_rep, 0.55)
        scorefxn.set_weight(ScoreType.fa_intra_rep, 0.005)
        scorefxn.set_weight(ScoreType.fa_sol, 1.0)
        scorefxn.set_weight(ScoreType.lk_ball_wtd, 1.0)
        scorefxn.set_weight(ScoreType.fa_intra_sol_xover4, 1.0)
        scorefxn.set_weight(ScoreType.fa_elec, 1.0)
        scorefxn.set_weight(ScoreType.hbond_lr_bb, 1.0)
        scorefxn.set_weight(ScoreType.hbond_sr_bb, 1.0)
        scorefxn.set_weight(ScoreType.hbond_bb_sc, 1.0)
        scorefxn.set_weight(ScoreType.hbond_sc, 1.0)
        scorefxn.set_weight(ScoreType.dslf_fa13, 1.25)
        scorefxn.set_weight(ScoreType.rama_prepro, 0.45)
        scorefxn.set_weight(ScoreType.p_aa_pp, 0.4)
        scorefxn.set_weight(ScoreType.fa_dun, 0.7)
        scorefxn.set_weight(ScoreType.omega, 0.6)
        scorefxn.set_weight(ScoreType.pro_close, 1.25)
        scorefxn.set_weight(ScoreType.yhh_planarity, 0.625)
        scorefxn.set_weight(ScoreType.ref, 1.0)

        return [
            scorefxn.score_by_scoretype(pose, ScoreType.fa_atr),
            scorefxn.score_by_scoretype(pose, ScoreType.fa_rep),
            scorefxn.score_by_scoretype(pose, ScoreType.fa_intra_rep),
            scorefxn.score_by_scoretype(pose, ScoreType.fa_sol),
            scorefxn.score_by_scoretype(pose, ScoreType.lk_ball_wtd),
            scorefxn.score_by_scoretype(pose, ScoreType.fa_intra_sol_xover4),
            scorefxn.score_by_scoretype(pose, ScoreType.fa_elec),
            scorefxn.score_by_scoretype(pose, ScoreType.hbond_lr_bb),
            scorefxn.score_by_scoretype(pose, ScoreType.hbond_sr_bb),
            scorefxn.score_by_scoretype(pose, ScoreType.hbond_bb_sc),
            scorefxn.score_by_scoretype(pose, ScoreType.hbond_sc),
            scorefxn.score_by_scoretype(pose, ScoreType.dslf_fa13),
            scorefxn.score_by_scoretype(pose, ScoreType.rama_prepro),
            scorefxn.score_by_scoretype(pose, ScoreType.p_aa_pp),
            scorefxn.score_by_scoretype(pose, ScoreType.fa_dun),
            scorefxn.score_by_scoretype(pose, ScoreType.omega),
            scorefxn.score_by_scoretype(pose, ScoreType.pro_close),
            scorefxn.score_by_scoretype(pose, ScoreType.yhh_planarity),
            scorefxn.score_by_scoretype(pose, ScoreType.ref)
        ]

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
        self.mutate()


def main():
    psept = Psept()
    psept.load_savs('test_sav.txt')
    psept.scoring()


if __name__ == '__main__':
    main()
