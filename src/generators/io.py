import numpy as np
from .core import Generator

class CSV_Generator(Generator):

    def __init__(self, filename, delimiter=",", skip_header=0, skip_footer=0, verbose=True, name="data"):
        self.filename = filename
        self.delimiter = delimiter
        self.skip_header = skip_header
        self.skip_footer = skip_footer
        self.verbose = verbose
        self.name = name

    def is_mimo(self, x):
        if len(x) > 1:
            output = True
        else:
            output = False
        return output

    def show_info(self, data, index):
        N_t, N = data.shape
        print("# Generator: file loaded successfully")
        print("- Number of data: {}".format(N_t))
        print("- Data Length: {}".format(N))
        print("- Index selected: {}".format(index))

    def forward(self, x):
        index = x
        data = np.genfromtxt(self.filename, delimiter=self.delimiter, skip_header=self.skip_header, skip_footer=self.skip_footer)

        if self.verbose:
            self.show_info(data,index)

        if self.is_mimo(x):
            output = np.transpose(data[:,index])
        else:
            output = np.ravel(data[:,index])

        return output

class CSV_Generator_DP(Generator):

    def __init__(self, filenames, delimiter=",", skip_header=0, skip_footer=0, verbose=True, name="data"):
        self.filenames = filenames
        self.delimiter = delimiter
        self.skip_header = skip_header
        self.skip_footer = skip_footer
        self.verbose = verbose
        self.name = name

    def get_data_from_file(self):
        data_real_imag = np.genfromtxt(self.filenames[0], delimiter=self.delimiter, skip_header=self.skip_header, skip_footer=self.skip_footer)

        N = data_real_imag.shape[0]
        data = np.zeros((2, N), dtype=complex)
        data[0,:] = data_real_imag[:,0]+1j*data_real_imag[:,1]
        data[1,:] = data_real_imag[:,2]+1j*data_real_imag[:,3]
        return data

    def get_data_from_files(self):
        data_X_real_imag = np.genfromtxt(self.filenames[0], delimiter=self.delimiter, skip_header=self.skip_header, skip_footer=self.skip_footer)
        data_Y_real_imag = np.genfromtxt(self.filenames[1], delimiter=self.delimiter, skip_header=self.skip_header, skip_footer=self.skip_footer)

        N = data_X_real_imag.shape[0]
        data = np.zeros((2, N), dtype=complex)
        data[0,:] = data_X_real_imag[:,0]+1j*data_X_real_imag[:,1]
        data[1,:] = data_Y_real_imag[:,0]+1j*data_Y_real_imag[:,1]
        return data

    def show_info(self, data):
        N_t, N = data.shape
        print("# Generator: file loaded successfully")
        print("- Number of data: {}".format(N_t))
        print("- Data Length: {}".format(N))

    def forward(self, x):
        if len(self.filenames)==1:
            data = self.get_data_from_file()
        else:
            data = self.get_data_from_files()

        if self.verbose:
            self.show_info(data)

        return data