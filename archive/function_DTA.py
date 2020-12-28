# DTA Toolkit
#
# A data transformation algebra Toolkit for improved
# data cleaning provenance
#
#  Lan Li, Santiago Nunez-Corrales
# Supervisor: Bertram Ludaescher, CIRSS-iSchool

# Separation of concerns
#
# 1. How to represent datasets
#   1.1. How to decouple our model from the usual table/db model
#       - Only using Pandas to load and save data
#           * read_csv, write_csv
#           * structure S to Pandas dataframe
#       - Introspection:
#           * the ability to see all components of a dataset
#           * obtain individual values
#           * obtain multiple values
#           * obtain the signature of its corresponding ds space
# 2. How to represent and construct data transformations
#   2.1. How to construct cell transformations (Eq. 17)
#   2.2. How to use cell transformations to construct dataset transformations (Eq. 18)
#   2.3. Testing transformations
#       * Signature
#       * Is Delta an identity? For which dataset space (a.k.a. do signature match)? (Eq. 14)

import csv
from collections import Counter

import numpy as np
import pandas as pd


class Transformation:
    # def __init__(self, sig=None, phi=None,
    #                 pi=None, p=None, restrictions=None):
    #     self.signature = sig
    #     self._cell_impl = self._cell_transformation(phi, pi, p)
    #     self._implementation = self._ds_transformation(self.cell_impl, restrictions)
    #     self.is_identity = False

    # Eq. 12
    def __init__(self, dataset):
        self.signature = dataset.signature
        self._cell_impl = self._cell_transformation(self._identity_phi,
                                                    self._identity_pi,
                                                    self._identity_p)
        self._implementation = self._ds_transformation(self._cell_impl,
                                                        (dataset.I, dataset.J))
        self.is_identity = True

    # (c, (i,j))
    def _cell_transformation(self, phi, pi, p):
        return 0

    def _ds_transformation(self, cell_trans, restrictions):
        return 0

    def apply(self, dataset):
        if self._test_signature(dataset):
            pass
        else:
            # Raise error
            pass

    def _test_signature(self, dataset):
        if self.signature == dataset.signature:
            return True
        else:
            # Raise an error: incompatible signatures for transformation
            return False

    # Eq. (13)
    def compose(self, other):
        # TODO: implement this first
        # What needs to happen:
        #   * 
        pass

    # Additional helper functions
    @staticmethod
    def _identity_phi(content):
        return content

    @staticmethod
    def _identity_pi(i, j):
        return i, j
    
    @staticmethod
    def _identity_p(c):
        return True

    def pd_csv(self,data_p):
        self.D = pd.read_csv(data_p)

    def base_del_row_op(self,row_idx):
        '''
        # copy row (value + position)
        delete row (position)
        # add row (position)
        input dataset
        :return: output dataset/ stringIO / List
        '''
        self.D = self.D.drop(row_idx)

    def base_del_col_op(self,drop_col):
        ''' A (COUNT) B () C ()
        A:3 B:5 C:7
        质数相加，唯一性？
        add column
        # copy column
        delete column
        # rename column
        # inter-column: date format/ number/ text...
        # move column: copy + position & delete
        :return:
        '''
        self.D = self.D.drop(columns=drop_col)

    def base_add_col_op(self,new_col,old_col,insert_idx,copy=True,*add_col_val):
        # new column name
        # old column name
        # new column position
        if copy:
            # copy the value from the old column
            new_col_val=self.D[old_col]
            self.D.insert(loc=insert_idx, column=new_col, value=new_col_val)
        else:
            # new values based on the old column
            for arg in add_col_val:
                self.D.insert(loc=insert_idx,column=new_col, value=arg)

    def move_col_op(self,insert_idx,old_idx,new_col,old_col):
        # insert_idx: move the column to new position
        # old_idx: delete the old position column
        # new_col: new name for moving column
        # old_col: old position column name
        # copy add
        self.base_add_col_op(new_col,old_col,insert_idx,copy=True)
        # delete
        self.base_del_col_op(old_idx)

    def split_col_op(self,old_col,regex=',',keep_old=True):
        # split
        new_df = pd.concat(self.D[old_col].str.split(regex, expand=True), axis=1, keys=self.D.columns)
        new_df.columns = new_df.columns.map(lambda x: '_'.join((x[0], str(x[1]))))
        new_df = new_df.replace({'': np.nan, None: np.nan})
        if keep_old:
            # the whole table + new generated columns
            self.D = pd.concat([self.D, new_df],axis=1)
            # self.base_add_col_op()
            # self.D[]=self.D[]
        else:
            # the whole table - old_col +new generated columns
            self.base_del_col_op(old_col)
            self.D = pd.concat([self.D, new_df], axis=1)

    def cell_pos(self):
        # cell index
        pass

    def split_simplify_op(self):
        # 对于任何一个operation， 首先进行拆分简化，通过排序来比较路径是否相等
        # 任何OPERATION调用他以后都能得到一个简化版的细分路径/BASE OPS
        # reverse narrow: 哪些操作是相反的
        # column index: 这俩是不是作用在同一个DOMAIN
        # 动态规划； 概率方式
        # add minus

        pass

    def topology_(self):
        pass


class Dataset:
    '''
    存储数据，只做内部转换
    函数写外面，2个DTA??
    发现外部过于Messy
    '''
    def __init__(self):
        # 传入数据集，
        # five parameters for dataset
        self.R = dict() # regular expression for each column
        self.C = dict() # (content, frequency)
        self.I = set() # row set
        self.J = set() # column set
        self.S = dict() # element from C -> pairs from I x J
        self.data_space=set() # current dataset

    def equivalent(self, dataset):
        pass

    def equal(self, dataset):
        pass

    def data_regex(self,data_p):
        return self.R

    def data_content(self,data_p):
        data = []
        with open(data_p, 'r')as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader) # remove header
            for row in csvreader:
                data += row
        '''
        L:  {(value,frequency),...}
        series.value_counts
        '''
        self.L=Counter(data)

    def row_column_index(self,data_p):
        df=pd.read_csv(data_p)
        row_length=df.shape[0]
        column_length=df.shape[1]
        self.I=set(range(row_length))
        self.J=set(range(column_length))

    def structuring_func(self,data_p):
        # S: {c -> (i,j)}
        # self.S = np.cross(self.I, self.J)
        data = np.genfromtxt(data_p, dtype=None, delimiter=',', encoding='utf-8')
        print(data)
        for index, x in np.ndenumerate(data):
            self.S[x] = index

    def dataset_space(self):
        self.D = (self.R, self.L, self.S, self.I, self.J)

    # Helper function

    # CSV file
    # -> load using Pandas (DataFrame)
    # -> create new dataset (our model: R, C, I, J, S)
    # -> trasformations
    # ...
    # -> transformations
    # -> migrate our model to a DataFrame
    # -> save DataFrame to csv
    def _csv_to_dataset(self):
        pass

    def _dataset_to_dframe(self):
        pass


class DTAToolkit:
    def __init__(self, filename, savecurrent=False):
        self._source_file = filename
        self._trans_history = []
        self._save_curr = savecurrent
        self._most_recent = -1
        self._prev = None
        self._curr = None

    def signature_set(self,prev_data_sp, cur_data_sp):
        # return difference in data spaces of ops-
        # prev_data_sp: previous data space
        # prev_dsp_list: previous data space list
        # cur_data_sp: current data space
        # cur_dsp_list: current data space list
        # sig_set: signature set, unifying data space states changes from previous to current
        sig_set = []
        prev_dsp_list = []
        cur_dsp_list = []
        for i in range(len(prev_data_sp)):
            # pair = []
            if prev_data_sp[i] != cur_data_sp[i]:
                prev_dsp_list.append(prev_data_sp[i])
                cur_dsp_list.append(cur_data_sp[i])
        sig_set += (prev_dsp_list, cur_dsp_list)

        return sig_set

    def load_prev(self, filename):
        pass

    def switch_save_mode(self):
        self._save_curr = not self._save_curr

    def add_transformation(self, delta):
        self._trans_history.append(delta)

    def apply_history(self):
        return self.apply_to_kth_history(len(self._trans_history))
        pass

    def apply_to_kth_history(self, k):
        if k < 0:
            return self._prev
        elif k >= len(self._trans_history):
            composition = self._compose(0, len(self._trans_history))
            ds = composition.apply(self._prev)

            self._most_recent = len(self._trans_history) - 1

            if self._save_curr:
                self._curr = ds

            return ds
        else:
            composition = self._compose(0, k)
            ds = composition.apply(self._prev)
            
            self._most_recent = k

            if self._save_curr:
                self._curr = ds

            return ds

    def step(self):
        pass

    def _compose(self, i, j):
        return

