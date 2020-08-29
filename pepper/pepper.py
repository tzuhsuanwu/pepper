from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
import re
import os
import time
import logging
import datetime
from urllib.request import urlretrieve
from urllib.error import HTTPError
from urllib.error import URLError
from pyrosetta import init
from pyrosetta.io import pose_from_pdb
from pyrosetta.toolbox.cleaning import cleanATOM
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.rosetta.core.scoring import ScoreFunction
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.core.scoring import calc_total_sasa
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


class Pepper:
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

    def __init__(self, remove_pdb_after_scoring=True, pdb_info_path='pdb_info.json'):
        """
        Initilize and setting for Pepper
        """
        
        dt = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        self.__start_time = dt
        self.log_initialize()
        self.__main_log.info('Program start')
        self.__data = None
        self.__module_root_path = Path(__file__).parent
        self.__remove_pdb_after_scoring = remove_pdb_after_scoring

        try:
            self.__main_log.info('Loading pdb_info.json...')
            with open(self.__module_root_path / pdb_info_path, 'r') as f:
                self.__pdb_info = json.load(f)
            self.__main_log.info('Load success')
        except FileNotFoundError:
            self.__main_log.warning('pdb_info.json not found. Create a empty json file.')
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
        self.__max_pdb_size = None
        self.__min_pdb_size = None

    def log_initialize(self):
        """
        Log initialize and format setting
        """
        logging.getLogger('').handlers = []
        logging.basicConfig(
            format='%(asctime)s %(name)s %(levelname)s %(message)s ',
            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S'
        )
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        main_log = logging.getLogger('main')
        fh = logging.FileHandler(f'{self.__start_time}.log')
        fh.setFormatter(formatter)
        main_log.addHandler(fh)
        self.__main_log = main_log

    def load_savs(self, filepath):
        """
        Load input sav file
        Input format: {UniProt ID} {mutated position} {wildtype amino acid} {mutated amino acid}
        e.g., P04637 35 L F
        """
        try:
            self.__main_log.info('Loading input file...')
            with open(Path(filepath)) as f:
                data = f.read().splitlines()
        except:
            self.__main_log.warning('Load failed. Program exit.')
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

    def set_max_pdb_size(self, max_size):
        """
        Set max PDB size. All PDB which size is larger than this threshold will be excluded.
        """
        self.__max_pdb_size = max_size

    def set_min_pdb_size(self, min_size):
        """
        Set min PDB size. All PDB which size is smaller than this threshold will be exclueded.
        """
        self.__min_pdb_size = min_size

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
        self.__main_log.info(f'Start getting the size of {pdb}')
        full_pdb_path = self.__root_pdb_folder / 'full_pdb' / (pdb + '.pdb')
        clean_pdb_path = self.__root_pdb_folder / 'clean_pdb' / (pdb + '.clean.pdb')
        if not os.path.isfile(clean_pdb_path):
            download_success = False
            for i in range(10):
                try:
                    self.download_pdb(pdb)
                    download_success = True
                    break
                except HTTPError:
                    self.__main_log.warning(f'Download {pdb} failed')
                    self.__pdb_info['download_failed']['data'].append(pdb)
                    os.remove(full_pdb_path)
                    return None
                except URLError:
                    time.sleep(5)
                    continue
            if not download_success:
                self.__main_log.warning(f'Downloading {pdb} failed')
                self.__pdb_info['download_failed']['data'].append(pdb)
                os.remove(full_pdb_path)
                return None
            is_multimodle = self.check_pdb_is_multimodel(full_pdb_path)
            if is_multimodle:
                self.__main_log.warning(f'{pdb} is multimodel')
                self.__pdb_info['multimodel']['data'].append(pdb)
                os.remove(full_pdb_path)
                return None
            cleanATOM(str(full_pdb_path), str(clean_pdb_path))
        try:
            pose = pose_from_pdb(str(clean_pdb_path))
        except RuntimeError:
            self.__main_log.warning(f'Load {pdb} failed')
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
        self.__main_log.info('Start creating the ordered PDB list')
        self.__main_log.info(f'Total number of SAVs from input file: {len(self.__data)}')
        count = 1
        for sav in self.__data:
            self.__main_log.debug(f'Current sav: {sav}')
            uniprot, pos = sav.split()[:2]
            pos = int(pos)
            pdb_size_list = []

            # Get pdb list for sav gene and process each pdb.
            try:
                pdb_list = self.__pdb_info['uniprot_pdb']['data'][uniprot]
            except KeyError:
                self.__main_log.warning(f'Create failed ({sav}): No related PDB are found.')
                self.__sav_to_ordered_pdb.append([sav, 'No related PDB are found.'])
                continue

            self.__main_log.debug(f'pdb_list: {pdb_list}')
            for pdb in pdb_list:

                # Continue if pdb in skip list.
                if pdb in self.__skip_pdb_list:
                    self.__main_log.debug(f'{pdb} is in skip_list')
                    continue

                # Find corresponding dataframe.
                temp_df = self.__map_df.loc[(self.__map_df['PDB'] == pdb) & (self.__map_df['SP_PRIMARY'] == uniprot)]
                # Filter data which wasn't containing mutated position.
                temp_df = temp_df.loc[(self.__map_df['SP_BEG'] < pos) & (self.__map_df['SP_END'] > pos)]
                if temp_df.empty:
                    self.__main_log.debug(f'PDB skip: Mutated position is not in {pdb}')
                    continue

                pdb_series = temp_df.iloc[0]
                chain = pdb_series['CHAIN']
                pdb_beg = int(re.sub('[^0-9]', '', pdb_series['PDB_BEG']))
                pdb_end = int(re.sub('[^0-9]', '', pdb_series['PDB_END']))
                uni_beg = int(pdb_series['SP_BEG'])
                uni_end = int(pdb_series['SP_END'])

                # (pdb_end - pdb_beg) != (uni_end - uni_beg) indicate that wrong positions were recorded.
                if (pdb_end - pdb_beg) != (uni_end - uni_beg):
                    self.__main_log.debug(f'PDB skip: Wrong position were recorded in {pdb}')
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
                self.__main_log.warning(f'Create faile ({sav}): No PDB are usable after create candidate pdb list.')
                self.__sav_to_ordered_pdb.append([sav, 'No PDB are usable after create candidate pdb list.'])
                continue

            # Sort by size and order by median.
            sort_by_size = sorted(pdb_size_list, key=lambda x: x[1])
            middle_start_list = []
            while sort_by_size:
                median = len(sort_by_size)//2
                middle_start_list.append(sort_by_size.pop(median))
            self.__main_log.info(f'Create success {sav} ({count} / {len(self.__data)})')
            self.__sav_to_ordered_pdb.append([sav, middle_start_list])
            count += 1

    def mutate_and_scoring(self):
        """
        Mutate each SAVs of input file
        """
        self.__main_log.info('Start mutate and scoring')
        # Initialize and create result dataframe
        result_df_column = [
            'SAV', 'predict_status', 'predict_label', 'ordered_PDB_list', 'ordered_list_length',
            'current_PDB', 'current_PDB_info', 'chain', 'position', 'PDB_size', 'chain_length',
        ]
        result_df_column = result_df_column + self.__whole_column + ['wt_SASA', 'mu_SASA', 'diff_SASA']
        result_df = pd.DataFrame(columns=result_df_column)
        result_df['SAV'] = [i[0] for i in self.__sav_to_ordered_pdb]
        result_df['ordered_PDB_list'] = [i[1] for i in self.__sav_to_ordered_pdb]
        loop_round = 0
        while True:
            loop_round += 1
            # Record "failed" if no more PDB in candidate list of a SAV
            result_df.loc[(result_df['ordered_list_length'] == 0) & (result_df['predict_status'] == 'unfinished'), 'predict_status'] = 'failed'

            # Extract sub_df of unfinished SAVs
            sub_df = result_df[~result_df.predict_status.isin(['success', 'failed'])]

            if sub_df.empty:
                self.__result = result_df
                break

            # Pop new PDB from ordered_PDB_list and update the information
            sub_df['current_PDB_info'] = sub_df['ordered_PDB_list'].map(lambda x: x[0])
            sub_df['current_PDB'] = sub_df['current_PDB_info'].map(lambda x: x[0])
            sub_df['PDB_size'] = sub_df['current_PDB_info'].map(lambda x: x[1])
            sub_df['chain'] = sub_df['current_PDB_info'].map(lambda x: x[2])
            sub_df['position'] = sub_df['current_PDB_info'].map(lambda x: x[3])
            sub_df['ordered_PDB_list'] = sub_df['ordered_PDB_list'].apply(lambda x: x[1:])
            sub_df['ordered_list_length'] = sub_df['ordered_PDB_list'].apply(lambda x: len(x))
            count = 0
            for pdb in sub_df['current_PDB'].unique():
                count += 1
                self.__main_log.info(f'Current progress: {count} / {len(sub_df["current_PDB"].unique())}, Round: {loop_round}')
                # Check if PDB file is exist
                pdb_usable = False

                # Download PDB if not exist
                if not os.path.isfile(self.__clean_pdb_folder / (pdb + '.clean.pdb')):
                    for i in range(10):
                        try:
                            self.download_pdb(pdb)
                            break
                        except HTTPError:
                            break
                        except URLError:
                            time.sleep(5)
                            continue
                    cleanATOM(str(self.__full_pdb_folder / (pdb + '.pdb')), str(self.__clean_pdb_folder / (pdb + '.clean.pdb')))
                try:
                    # Calculate wildtype PDB score
                    wt_pose = pose_from_pdb(str(self.__clean_pdb_folder / (pdb + '.clean.pdb')))
                    wt_score = np.array(self.pyrosetta_scoring(wt_pose))
                    wt_sum = np.sum(wt_score)
                    wt_score = np.append(wt_score, wt_sum)
                    pdb_usable = True
                except RuntimeError:
                    pass

                temp_df = sub_df[sub_df['current_PDB'] == pdb]
                for idx, row in temp_df.iterrows():
                    # If PDB is not usable, skip
                    if not pdb_usable:
                        self.__main_log.debug(f'({temp_df.loc[idx]["SAV"]}) PDB {pdb} not usable, label as unfinished.')
                        sub_df.at[idx, 'predict_status'] = 'unfinished'
                        continue
                    chain = row['chain']
                    position_in_pdb = row['position']
                    wt_aa, mu_aa = row['SAV'].split()[2:4]

                    # Get actual mutate position in pose
                    position_in_pose = wt_pose.pdb_info().pdb2pose(chain, int(position_in_pdb))

                    # If position not found or residue in pose not equal to SAV, continue
                    if position_in_pose == 0 or wt_pose.residue(position_in_pose).name1() != wt_aa:
                        self.__main_log.debug(f'({temp_df.loc[idx]["SAV"]}) position not found or residue in pose not equal to SAV, label as unfinished.')
                        sub_df.at[idx, 'predict_status'] = 'unfinished'
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
                    sub_df.at[idx, 'predict_status'] = 'success'
                    sub_df.at[idx, 'chain_length'] = chain_length
                    # Calculate score of SASA
                    wt_sasa = calc_total_sasa(wt_pose, self.__probe_radius)
                    mu_sasa = calc_total_sasa(mutant_pose, self.__probe_radius)
                    sub_df.at[idx, 'wt_SASA'] = wt_sasa
                    sub_df.at[idx, 'mu_SASA'] = mu_sasa
                    sub_df.at[idx, 'diff_SASA'] = mu_sasa - wt_sasa

                    for i in range(len(self.__whole_column)):
                        sub_df.at[idx, self.__whole_column[i]] = result[i]
                    self.__main_log.info(f'{temp_df.loc[idx]["SAV"]} successful scoring')
                result_df.update(sub_df)
                result_df.to_csv('structure_energies_score.csv', index=False)
                if self.__remove_pdb_after_scoring:
                    os.remove(self.__root_pdb_folder / 'full_pdb' / (pdb + '.pdb'))
                    os.remove(self.__root_pdb_folder / 'clean_pdb' / (pdb + '.clean.pdb'))

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
        self.__main_log.info('Loading uniprot_segments_observed.csv...')
        try:
            self.__map_df = pd.read_csv(self.__module_root_path / 'uniprot_segments_observed.csv', skiprows=1)
        except FileNotFoundError:
            self.__main_log.warning('Load failed. File not found.')
            raise FileNotFoundError('File not found. Please put the file in this folder and try again.')

        # Generate ordered pdb list for input savs.
        self.create_ordered_pdb_list()

        # Mutate and score SAVs
        self.mutate_and_scoring()

    def prediction(self):
        """
        Predict input SAVs using pre-trained lgbm model
        """
        features = self.__whole_column + ['wt_SASA', 'mu_SASA', 'diff_SASA']
        # Load pre-trained model
        with open(self.__module_root_path / 'lgbm_prediction_model.pkl', 'rb') as f:
            clf = pickle.load(f)

        with open(self.__module_root_path / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open(self.__module_root_path / 'selector.pkl', 'rb') as f:
            selector = pickle.load(f)

        # Feature engineering
        # High correlation features will be divided by PDB size
        divided_by_pdb_size_columns = [
            "wt_fa_atr", "wt_fa_intra_rep", "wt_fa_sol", "wt_fa_elec", "wt_fa_dun", "wt_ref",
            "mu_fa_atr", "mu_fa_intra_rep", "mu_fa_sol", "mu_fa_elec", "mu_fa_dun", "mu_ref",
            "wt_SASA", "mu_SASA"
        ]
        for columns in divided_by_pdb_size_columns:
            self.__result[columns] = self.__result[columns].divide(self.__result['PDB_size'])

        # Scale and selection
        temp_df = self.__result.drop(columns=['predict_label'])
        temp_df = temp_df.dropna()
        not_nan_idx = temp_df.index
        X_test = temp_df[features]
        X_test = scaler.transform(X_test)
        X_test = selector.transform(X_test)

        # Prediction
        y_pred = clf.predict(X_test)
        for i in range(len(y_pred)):
            self.__result.loc[not_nan_idx[i]]['predict_label'] = y_pred[i]

        self.__result.to_csv('predict_result.csv', index=False)
