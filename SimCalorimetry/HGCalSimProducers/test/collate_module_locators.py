import pandas as pd

input_files = [
    'P5/maps/modulelocator_P5v34.txt',
    'P5/maps/modulelocator_P5v35.txt',
    'P5/maps/modulelocator_P5v36.txt',
    'P5/maps/modulelocator_P5v37.txt',
    'P5/maps/modulelocator_P5v38.txt'
]

data = [ pd.read_csv(f'HGCalCommissioning/Calibrations/{f}', header='infer', sep=r'\s+') for f in input_files ]
data = pd.concat(data,ignore_index=True)

def _reassignUV(row):
    cassette = row['cassette']
    u = row['u']
    v = row['v']
    for i in range(cassette-1):
        newu = u-v
        newv = u
        u = newu
        v = newv
    return u,v
data['u'], data['v'] = zip( *data.apply( _reassignUV, axis=1) )
data['becrate'] = data['cassette']
data['zside'] = 1

data.to_csv(
    'HGCalCommissioning/Calibrations/P5/maps/modulelocator_C01toC05.txt',
    sep=' ',
    index=False
)

