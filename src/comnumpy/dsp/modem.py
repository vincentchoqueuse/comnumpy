import os.path as path
import numpy as np
from .core import Processor
from .functional import hard_projector

def get_alphabet(modulation, order, type="gray", is_real=False, norm=True):
    # extract alphabet
    pathname = path.dirname(path.abspath(__file__))
    filename = "{}/data/{}_{}_{}.csv".format(pathname, modulation, order, type)
    data = np.loadtxt(filename,delimiter=',',skiprows=1)
    alphabet = data[:,1]+1j*data[:,2]

    if is_real:
        alphabet = np.real(alphabet)

    if norm == True :
        alphabet = alphabet/np.sqrt(np.mean(np.abs(alphabet)**2))

    return alphabet


def sym_2_bin(sym,width=4):

    data = []
    for indice in range(len(sym)):
        data.append(np.binary_repr(sym[indice],width))

    string = ''.join(data)

    return np.array(list(string), dtype=int)


class Modulator(Processor):

    def __init__(self,alphabet, name="modulator"):
        super().__init__()
        self._alphabet = alphabet
        self.name = name

    def get_alphabet(self):
        return self._alphabet

    def forward(self,x):
        Y = self._alphabet[x]
        return Y


class Demodulator(Processor):

    def __init__(self,alphabet, name="demodulator"):
        super().__init__()
        self._alphabet = alphabet
        self.name = name

    def forward(self,x):
        s, x = hard_projector(x, self._alphabet)
        return s
    

class Str_2_Bin(Processor):
    """
    A processor for converting a string to a binary representation 
    as a NumPy array of 1s and 0s.

    Parameters
    ----------
    encoding : str, optional
        The character encoding used to convert the string to bytes.
        Default is 'utf-8'.

    Attributes
    ----------
    encoding : str
        The character encoding used for string-to-byte conversion.

    Methods
    -------
    forward(x)
        Converts a string to a NumPy array of 1s and 0s representing its binary form.

    """

    def __init__(self, encoding='utf-8', name="str2bin"):
        self.encoding = encoding
        self.name = name

    def forward(self, x):
        # Encode the string into bytes and then to a binary string
        byte_array = x.encode(self.encoding)
        binary_string = ''.join([bin(byte)[2:].zfill(8) for byte in byte_array])

        # Convert the binary string to a NumPy array of integers
        binary_np_array = np.array([int(bit) for bit in binary_string])
        return binary_np_array


class Bin_2_Str(Processor):
    """
    A processor for converting a NumPy array of 1s and 0s back to a string.

    Parameters
    ----------
    encoding : str, optional
        The character encoding used to decode the bytes back to a string.
        Default is 'utf-8'.

    Attributes
    ----------
    encoding : str
        The character encoding used for byte-to-string conversion.

    Methods
    -------
    forward(x)
        Converts a NumPy array of 1s and 0s back to a string.

    """

    def __init__(self, encoding="utf-8", name="bin2str"):
        self.encoding = encoding
        self.name = name

    def forward(self, x):
        # Convert the NumPy array to a binary string
        binary_string = ''.join(x.astype(str))

        # Split the binary string into chunks of 8 bits and convert to bytes
        byte_values = [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]
        byte_string = bytes(byte_values)

        # Decode the byte string to a UTF-8 string
        return byte_string.decode(self.encoding, errors='ignore')


class Bin_2_Data(Processor):
    """
    Class to convert a binary sequence (numpy array of 0s and 1s) into 
    data by grouping the sequence into blocks of N bits and converting 
    each block to an integer. Supports both MSB-first and LSB-first orderings.

    Attributes
    ----------
    block_size : int
        The size of each block of bits to be converted into an integer.
    bit_order : str
        The bit order for conversion, either 'msb' or 'lsb'.

    Methods
    -------
    forward(binary_sequence)
        Converts a binary sequence into data by grouping into blocks of N bits.
    """

    def __init__(self, block_size, bit_order='msb', name="bin2data"):
        self.block_size = block_size
        self.bit_order = bit_order.lower()
        if self.bit_order not in ['msb', 'lsb']:
            raise ValueError("bit_order must be either 'msb' or 'lsb'.")
        self.name = name

    def forward(self, x):
        length = len(x)
        if length % self.block_size != 0:
            raise ValueError("Length of binary sequence must be a multiple of block_size.")

        # Reshape the array and convert each block to an integer
        reshaped = x.reshape(-1, self.block_size)
        if self.bit_order == 'lsb':
            reshaped = np.fliplr(reshaped)
        y = np.array([int(''.join(map(str, bits)), 2) for bits in reshaped])

        return y
    

class Data_2_Bin(Processor):
    """
    Class to convert a sequence of integers back into a binary sequence 
    (numpy array of 0s and 1s) by converting each integer to a block of N bits. 
    Supports both MSB-first and LSB-first orderings.

    Attributes
    ----------
    block_size : int
        The size of each block of bits for each integer.
    bit_order : str
        The bit order for conversion, either 'msb' or 'lsb'.

    Methods
    -------
    forward(data_sequence)
        Converts a sequence of integers back into a binary sequence.
    """

    def __init__(self, block_size, bit_order='msb', name="data2bin"):
        self.block_size = block_size
        self.bit_order = bit_order.lower()
        if self.bit_order not in ['msb', 'lsb']:
            raise ValueError("bit_order must be either 'msb' or 'lsb'.")
        self.name = name

    def forward(self, x):
        binary_strings = [np.binary_repr(num, width=self.block_size) for num in x]
        binary_sequence = ''.join(binary_strings)
        
        if self.bit_order == 'lsb':
            binary_sequence = ''.join([binary[::-1] for binary in binary_strings])

        y = np.array(list(binary_sequence), dtype=int)
        return y
