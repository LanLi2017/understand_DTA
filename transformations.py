def Diff(li1, li2):
    return list(list(set(li1) - set(li2)) + list(set(li2) - set(li1)))


def intersection(cur_content, prev_content):
    # content: {signature: value, ...}
    res = {}
    for key,value in cur_content:
        prev_v = prev_content[key]
        if prev_v != value:
            res.update({key:value})
    return res


class Transformation:
    def __init__(self, cur_ds, prev_ds):
        self._cell_impl = dict()
        self._implementation = dict()
        self.rigid_trans = False # asynchronous change structure/content
        self.geo_trans = False # change structure,content,index
        self.iden_trans = False # identity transformation
        self.p = False # applied function reversible or not
        self.prov = None # if function is reversible, provenance is None
        self.cur_content = cur_ds.get_Metadata()[0]
        self.cur_Structure = cur_ds.get_Metadata()[1]
        self.cur_pi = cur_ds.get_Metadata()[2]
        self.prev_content = prev_ds.get_Metadata()[0]
        self.prev_Structure = prev_ds.get_Metadata()[1]
        self.prev_pi = prev_ds.get_Metadata()[2]

    def _cell_transformation(self):
        '''
        rigid cell transformation τ(c → (i, j))
        predicate p applies function ϕ to c ∈ C [content]
        function pi to pairs(i,j) [row, column]
        '''
        # p=True, irreversible (c → (i, j),ϕ1(c) → π1(i, j))
        self.identify_trans_type()
        self._identity_p()
        if self.rigid_trans:
            if self.p:
                self._cell_impl = self.cur_Structure
            else:
                self._cell_impl = self.prev_Structure
        elif self.geo_trans:
            pass
        else:
            pass

    def _ds_transformation(self):
        '''
        ∆|{ j}{i} = τ[S] ∩ {c → (i, j)|c ∈ C }
        '''
        self.identify_trans_type()
        self._identity_p()

        if self.rigid_trans:
            # signature will not alter
            self._cell_transformation()
            self._implementation = intersection(self.cur_content, self.prev_content)
            if self.p:
                self.prov = self._implementation
            else:
                self.prov = None
        elif self.geo_trans:
            pass
        return 0

    def _identity_p(self):
        # check if the phi is reversible or not
        pass

    def identify_trans_type(self):
        # check transformation type
        # TODO: regular expression ==> encoding transformation is
        #  also rigid transformation
        if intersection(self.cur_content, self.prev_content) \
                and Diff(self.prev_Structure, self.cur_Structure) \
                and Diff(self.prev_pi, self.cur_pi):
            self.geo_trans = True
        elif not intersection(self.cur_content, self.prev_content) \
                and not Diff(self.prev_Structure, self.cur_Structure) \
                and not Diff(self.prev_pi, self.cur_pi):
            # identity transformation
            self.iden_trans = True
        elif intersection(self.cur_content, self.prev_content) \
                and not Diff(self.prev_Structure, self.cur_Structure) \
                and not Diff(self.prev_pi, self.cur_pi):
            self.rigid_trans = True
        elif not intersection(self.cur_content, self.prev_content) \
                and Diff(self.prev_Structure, self.cur_Structure) \
                and not Diff(self.prev_pi, self.cur_pi):
            self.rigid_trans = True

