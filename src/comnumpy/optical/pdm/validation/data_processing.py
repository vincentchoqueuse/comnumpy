import numpy as np
import pandas as pd
import pathlib

n = 15
b_n = bin(n)

def get_quadrant():
    pass

data_path = 'src\\comnumpy\\core\\data'
folder = pathlib.Path(data_path)
for item in folder.iterdir():
    if item.is_file() and "_gray" in item.name:
        df = pd.read_csv(item)
        data = df.iloc[:,0].to_numpy()
        real = df.iloc[:,1].to_numpy()
        imag = df.iloc[:,2].to_numpy()
        new_column = []
        width = int(np.log2(len(data)))
        line_vector = np.ravel(data)
        for i in range(len(line_vector)):
            bin_val = np.binary_repr( int(line_vector[i]), width )
            new_column.append(bin_val)
        df['s'] = new_column
        df['real'] = real
        df['imag'] = imag
        new_path = item.with_name( item.name  )
        df.to_csv( new_path, index=False)