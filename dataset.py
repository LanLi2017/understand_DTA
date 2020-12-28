import pandas as pd


class Dataset:
    '''
    a dataset is a collection of elements with some algebraic structure
    quintuple (Regular expression, content, structure, row_I, column_J)
    Regular expression: mining each column and return corresponding regular expression
    content: (c, lambda): c is content, map to lambda=> 2^rowindex * 3^colindex
    Structure: content : (row, column)
    row_I: row index set
    column_J: column index set
    '''

    def __init__(self, file_path):
        self.fp = file_path
        self.df = None
        self.row_I = []
        self.column_J = []
        self.content = dict()
        self.Structure = []

    def read_ds(self, ds_format='csv'):
        """read dataset"""
        if ds_format == 'csv':
            self.df = pd.read_csv(self.fp)
        elif ds_format == 'h5':
            # HDF5 file stands for Hierarchical Data Format 5.
            self.df = pd.read_hdf(self.fp)
        elif ds_format == 'xlsx':
            self.df = pd.read_excel(self.fp)
        else:
            raise Exception('Unrecognized file format.')

    def get_index(self):
        row_length = self.df.shape[0]
        column_length = self.df.shape[1]
        self.row_I = list(range(row_length))
        self.column_J = list(range(column_length))

    def get_Metadata(self):
        signature = 0  # lambda
        column_list = list(self.df.columns.values)
        pi = list()  # pairs of (row index, column index )
        for value in self.df.itertuples():
            row_id = value[0]
            element = value[1:]
            for i, e in enumerate(element):
                pos = (row_id, i)  # coordinates of cell
                signature = pow(2, row_id) * pow(3, i)
                # self.content.append([e,signature])
                self.content.update({signature: e})
                self.Structure.append([e, pos])
                pi.append(pos)
        return self.content, self.Structure, pi

    def get_regex(self):
        # sherlock?
        #TODO
        pass
