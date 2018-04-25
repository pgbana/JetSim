import h5py

def or_data(file_name, *args, mode='r'):
    with h5py.File(file_name + '.hdf5', mode) as df:
        datas = []
        for arg in args:
            datas.append(df[arg][()])
        df.close()
    if len(args)==1:
        return datas[0]
    else:
        return datas

def ow_data(file_name, mode='r+', prefix='', **kwargs):
    with h5py.File(file_name + '.hdf5', mode) as sf:
        for key, value in kwargs.items():
            sf.create_dataset(prefix + key, data=value)

def create_file(file_name):
    with h5py.File(file_name + '.hdf5', 'w') as sf:
        sf.create_dataset('File_Name', data=file_name)