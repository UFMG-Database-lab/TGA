import numpy as np
import scipy.sparse as ss
from scipy.sparse import lil_matrix, csr_matrix, dok_matrix
from scipy._lib.six import xrange, zip
from scipy.sparse.sputils import isshape, isintlike

def swap(s1, s2):
    return s2, s1   
class slil_matrix(lil_matrix):
    def __init__(self, arg1, dtype=None):
        if isinstance(arg1, int) or isinstance(arg1,np.integer):
            lil_matrix.__init__(self, (arg1,arg1), dtype=dtype)
        elif isinstance(arg1,tuple) and isshape(arg1):
            lil_matrix.__init__(self, arg1, dtype=dtype)
        else:
            raise ValueError('invalid use of constructor. sll only use shape')
    def _param_to_index(self, i, j):
        is_int_i = isinstance(i, int) or isinstance(i, np.integer)
        is_list_i = isinstance(i, list) or isinstance(i, np.ndarray)

        is_int_j = isinstance(j, int) or isinstance(j, np.integer)
        is_list_j = isinstance(j, list) or isinstance(j, np.ndarray)

        if (is_list_i and is_list_j) or (is_int_i and is_int_j):
            return i,j
        
        is_slice_i = isinstance(i, slice)
        is_slice_j = isinstance(j, slice)
        if is_slice_i:
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else self.shape[0]
            step = i.step if i.step is not None else 1
            i = np.array(xrange(start,stop,step))
        if is_slice_j:
            start = j.start if j.start is not None else 0
            stop = j.stop if j.stop is not None else self.shape[1]
            step = j.step if j.step is not None else 1
            j = np.array(xrange(start,stop,step))
        if (is_list_i or is_slice_i) and (is_slice_j or is_list_j):
            new_i = []
            new_j = []
            for idx_i in i:
                for idx_j in j:
                    new_i.append(idx_i)
                    new_j.append(idx_j)
            i = new_i
            j = new_j
        if is_int_i:
            a = np.empty((len(j),), dtype=np.int)
            a.fill(i)
            i = a
        if is_int_j:
            a = np.empty((len(i),), dtype=np.int)
            a.fill(j)
            j = a
        return i,j

    def _get_shape_(self, i,j):
        is_int_i = isinstance(i, int) or isinstance(i, np.integer)
        if is_int_i:
            shape_i = 1
            
        is_int_j = isinstance(j, int) or isinstance(j, np.integer)
        if is_int_j:
            shape_j = 1
            
        is_slice_i = isinstance(i, slice)
        if is_slice_i:
            start = 0 if i.start is None else i.start
            stop = self.shape[0] if i.stop is None else i.stop
            shape_i = stop-start
            
        is_slice_j = isinstance(j, slice)
        if is_slice_j:
            start = 0 if j.start is None else j.start
            stop = self.shape[1] if j.stop is None else j.stop
            shape_j = stop-start
            
        is_list_j = isinstance(j, list) or isinstance(j, np.ndarray)
        is_list_i = isinstance(i, list) or isinstance(i, np.ndarray)
        if is_list_i and is_list_j:
            return 1, len(j)
        if is_list_i:
            shape_i = len(i)
        if is_list_j:
            shape_j = len(j)

        return shape_i, shape_j
    def __getitem__(self, index):
        """Return the element(s) index=(i, j), where j may be a slice.
        This always returns a copy for consistency, since slices into
        Python lists return copies.
        """

        # Scalar fast path first
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if ((isinstance(i, int) or isinstance(i, np.integer)) and
                    (isinstance(j, int) or isinstance(j, np.integer))):
                if j > i:
                    i,j = swap(i,j)
                return lil_matrix.__getitem__(self, (i,j))
        # Utilities found in IndexMixin
        i, j = self._unpack_index(index)
        #print("_unpack_index", i,j)
        shape_i, shape_j = self._get_shape_(i,j)
        #print("shape", shape_i, shape_j)
        i, j = self._param_to_index(i, j)
        #print("_param_to_index", i,j)
        i_ = []
        j_ = []
        for idx in range(len(i)):
            if j[idx] > i[idx]:
                ii,jj = swap(i[idx],j[idx])
                i_.append(ii)
                j_.append(jj)
            else:
                i_.append(i[idx])
                j_.append(j[idx])
        i = np.array(i_)
        j = np.array(j_)
        #print("normalized index", i, j)
        if i.size == 0:
            return lil_matrix(i.shape, dtype=self.dtype)
        return lil_matrix.__getitem__(self, (i,j)).reshape((shape_i, shape_j))
        
class scsr_matrix(csr_matrix):
    def __init__(self, arg1, shape=None, dtype=None):
        if shape is not None and isshape(shape):
            csr_matrix.__init__(self, arg1, dtype=dtype)
        elif isinstance(arg1, int) or isinstance(arg1,np.integer):
            csr_matrix.__init__(self, (arg1,arg1), dtype=dtype)
        elif isinstance(arg1,tuple) and isshape(arg1):
            csr_matrix.__init__(self, arg1, dtype=dtype)
        else:
            raise ValueError('invalid use of constructor. sll only use shape')
    def _param_to_index(self, i, j):
        is_int_i = isinstance(i, int) or isinstance(i, np.integer)
        is_list_i = isinstance(i, list) or isinstance(i, np.ndarray)

        is_int_j = isinstance(j, int) or isinstance(j, np.integer)
        is_list_j = isinstance(j, list) or isinstance(j, np.ndarray)

        if (is_list_i and is_list_j) or (is_int_i and is_int_j):
            return i,j
        
        is_slice_i = isinstance(i, slice)
        is_slice_j = isinstance(j, slice)
        if is_slice_i:
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else self.shape[0]
            step = i.step if i.step is not None else 1
            i = np.array(xrange(start,stop,step))
        if is_slice_j:
            start = j.start if j.start is not None else 0
            stop = j.stop if j.stop is not None else self.shape[1]
            step = j.step if j.step is not None else 1
            j = np.array(xrange(start,stop,step))
        if (is_list_i or is_slice_i) and (is_slice_j or is_list_j):
            new_i = []
            new_j = []
            for idx_i in i:
                for idx_j in j:
                    new_i.append(idx_i)
                    new_j.append(idx_j)
            i = new_i
            j = new_j
        if is_int_i:
            a = np.empty((len(j),), dtype=np.int)
            a.fill(i)
            i = a
        if is_int_j:
            a = np.empty((len(i),), dtype=np.int)
            a.fill(j)
            j = a
        return i,j

    def _get_shape_(self, i,j):
        is_int_i = isinstance(i, int) or isinstance(i, np.integer)
        if is_int_i:
            shape_i = 1
            
        is_int_j = isinstance(j, int) or isinstance(j, np.integer)
        if is_int_j:
            shape_j = 1
            
        is_slice_i = isinstance(i, slice)
        if is_slice_i:
            start = 0 if i.start is None else i.start
            stop = self.shape[0] if i.stop is None else i.stop
            shape_i = stop-start
            
        is_slice_j = isinstance(j, slice)
        if is_slice_j:
            start = 0 if j.start is None else j.start
            stop = self.shape[1] if j.stop is None else j.stop
            shape_j = stop-start
            
        is_list_j = isinstance(j, list) or isinstance(j, np.ndarray)
        is_list_i = isinstance(i, list) or isinstance(i, np.ndarray)
        if is_list_i and is_list_j:
            return 1, len(j)
        if is_list_i:
            shape_i = len(i)
        if is_list_j:
            shape_j = len(j)

        return shape_i, shape_j
    def __getitem__(self, index):
        """Return the element(s) index=(i, j), where j may be a slice.
        This always returns a copy for consistency, since slices into
        Python lists return copies.
        """

        # Scalar fast path first
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if ((isinstance(i, int) or isinstance(i, np.integer)) and
                    (isinstance(j, int) or isinstance(j, np.integer))):
                if j > i:
                    i,j = swap(i,j)
                return csr_matrix.__getitem__(self, (i,j))
        # Utilities found in IndexMixin
        i, j = self._unpack_index(index)
        #print("_unpack_index", i,j)
        shape_i, shape_j = self._get_shape_(i,j)
        #print("shape", shape_i, shape_j)
        i, j = self._param_to_index(i, j)
        #print("_param_to_index", i,j)
        i_ = []
        j_ = []
        for idx in range(len(i)):
            if j[idx] > i[idx]:
                ii,jj = swap(i[idx],j[idx])
                i_.append(ii)
                j_.append(jj)
            else:
                i_.append(i[idx])
                j_.append(j[idx])
        i = np.array(i_)
        j = np.array(j_)
        #print("normalized index", i, j)
        if i.size == 0:
            return csr_matrix(i.shape, dtype=self.dtype)
        return csr_matrix(csr_matrix.__getitem__(self, (i,j)).reshape((shape_i, shape_j)))

class sdok_matrix(dok_matrix):
    def __init__(self, arg1, dtype=None):
        if isinstance(arg1, int) or isinstance(arg1,np.integer):
            dok_matrix.__init__(self, (arg1,arg1), dtype=dtype)
        elif isinstance(arg1,tuple) and isshape(arg1):
            dok_matrix.__init__(self, arg1, dtype=dtype)
        else:
            raise ValueError('invalid use of constructor. sll only use shape')
    def _param_to_index(self, i, j):
        is_int_i = isinstance(i, int) or isinstance(i, np.integer)
        is_list_i = isinstance(i, list) or isinstance(i, np.ndarray)

        is_int_j = isinstance(j, int) or isinstance(j, np.integer)
        is_list_j = isinstance(j, list) or isinstance(j, np.ndarray)

        if (is_list_i and is_list_j) or (is_int_i and is_int_j):
            return i,j
        
        is_slice_i = isinstance(i, slice)
        is_slice_j = isinstance(j, slice)
        if is_slice_i:
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else self.shape[0]
            step = i.step if i.step is not None else 1
            i = np.array(xrange(start,stop,step))
        if is_slice_j:
            start = j.start if j.start is not None else 0
            stop = j.stop if j.stop is not None else self.shape[1]
            step = j.step if j.step is not None else 1
            j = np.array(xrange(start,stop,step))
        if (is_list_i or is_slice_i) and (is_slice_j or is_list_j):
            new_i = []
            new_j = []
            for idx_i in i:
                for idx_j in j:
                    new_i.append(idx_i)
                    new_j.append(idx_j)
            i = new_i
            j = new_j
        if is_int_i:
            a = np.empty((len(j),), dtype=np.int)
            a.fill(i)
            i = a
        if is_int_j:
            a = np.empty((len(i),), dtype=np.int)
            a.fill(j)
            j = a
        return i,j

    def _get_shape_(self, i,j):
        is_int_i = isinstance(i, int) or isinstance(i, np.integer)
        if is_int_i:
            shape_i = 1
            
        is_int_j = isinstance(j, int) or isinstance(j, np.integer)
        if is_int_j:
            shape_j = 1
            
        is_slice_i = isinstance(i, slice)
        if is_slice_i:
            start = 0 if i.start is None else i.start
            stop = self.shape[0] if i.stop is None else i.stop
            shape_i = stop-start
            
        is_slice_j = isinstance(j, slice)
        if is_slice_j:
            start = 0 if j.start is None else j.start
            stop = self.shape[1] if j.stop is None else j.stop
            shape_j = stop-start
            
        is_list_j = isinstance(j, list) or isinstance(j, np.ndarray)
        is_list_i = isinstance(i, list) or isinstance(i, np.ndarray)
        if is_list_i and is_list_j:
            return 1, len(j)
        if is_list_i:
            shape_i = len(i)
        if is_list_j:
            shape_j = len(j)

        return shape_i, shape_j
    def __getitem__(self, index):
        """Return the element(s) index=(i, j), where j may be a slice.
        This always returns a copy for consistency, since slices into
        Python lists return copies.
        """

        # Scalar fast path first
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if ((isinstance(i, int) or isinstance(i, np.integer)) and
                    (isinstance(j, int) or isinstance(j, np.integer))):
                if j > i:
                    i,j = swap(i,j)
                return dok_matrix.__getitem__(self, (i,j))
        # Utilities found in IndexMixin
        i, j = self._unpack_index(index)
        #print("_unpack_index", i,j)
        shape_i, shape_j = self._get_shape_(i,j)
        #print("shape", shape_i, shape_j)
        i, j = self._param_to_index(i, j)
        #print("_param_to_index", i,j)
        i_ = []
        j_ = []
        for idx in range(len(i)):
            if j[idx] > i[idx]:
                ii,jj = swap(i[idx],j[idx])
                i_.append(ii)
                j_.append(jj)
            else:
                i_.append(i[idx])
                j_.append(j[idx])
        i = np.array(i_)
        j = np.array(j_)
        #print("normalized index", i, j)
        if i.size == 0:
            return dok_matrix(i.shape, dtype=self.dtype)
        return dok_matrix.__getitem__(self, (i,j)).reshape((shape_i, shape_j))

    