import pandas as pd

input_files = [
    'P5/maps/modulelocator_P5v34.txt',
    'P5/maps/modulelocator_P5v35.txt',
    'P5/maps/modulelocator_P5v36.txt',
    'P5/maps/modulelocator_P5v37.txt',
    'P5/maps/modulelocator_P5v38.txt'
]

data = [ pd.read_csv(f'HGCalCommissioning/Calibrations/{f}', header='infer', sep=r'\s+') for f in input_files ]
data.append( data[-1].copy(deep=True) )
data[-1]['cassette']=6
data = pd.concat(data,ignore_index=True)

def _fixCopies(row):
    cassette = row['cassette']
    layer = row['plane']
    typecode = row['typecode']
    fedid = row['fedid']
    trig_fedid = row['trig_fedid']

    if cassette==5 and layer==26:
        typecode = typecode + '0'
    if cassette==6:
        typecode = typecode + '1'
        fedid=fedid+10
        trig_fedid=trig_fedid+10
    return typecode, fedid, trig_fedid

data['typecode'], data['fedid'], data['trig_fedid'] = zip(
    *data.apply( _fixCopies, axis=1)
)

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
data['crate'] = data['cassette']
data['zside'] = 1

data.to_csv(
    'HGCalCommissioning/Calibrations/P5/maps/modulelocator_C01toC06.txt',
    sep=' ',
    index=False
)

