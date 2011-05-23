autoSkim = {
    'MinimumBias':'MuonTrack+BeamBkg+ValSkim+LogError+HSCPSD',
    'ZeroBias':'LogError',
    'Commissioning':'MuonDPG+LogError',
    'Cosmics':'CosmicSP+LogError',
    'Mu' : 'WMu+ZMu+HighMET+LogError',    
    'EG':'WElectron+ZElectron+HighMET+LogError',
    'Electron':'WElectron+ZElectron+HighMET+LogError',
    'Photon':'WElectron+ZElectron+HighMET+LogError+DiPhoton+DoublePhoton',
    'JetMETTau':'LogError+DiJet+Tau',
    'JetMET':'HighMET+LogError+DiJet',
    'BTau':'LogError+Tau',
    'Jet':'HighMET+LogError+DiJet',
    'METFwd':'HighMET+LogError',

    'SingleMu' : 'WMu+ZMu+HighMET+LogError+HWW+DiTau+EXOLLRes+EXOMu',
    'DoubleMu' : 'WMu+ZMu+HighMET+LogError+HWW+EXOLLRes+EXOTriMu+EXODiMu',
    'SingleElectron' : 'WElectron+HighMET+LogError+HWW+Tau+EXOEle',
    'DoubleElectron' : 'ZElectron+LogError+HWW+EXOLLRes+EXOTriEle+EXODiEle',
    'MuEG' : 'LogError+HWW+EXOLLRes+EXO1E2Mu',
    'METBTag': 'HighMET+LogError',
    'MET': 'HighMET+LogError',

    'HT': 'HighMET+LogError',

    'Tau': 'LogError',
    'PhotonHad': 'LogError',
    'MuHad': 'LogError',
    'MultiJet': 'LogError',
    'MuOnia': 'LogError',
    'ElectronHad': 'LogError',
    'TauPlusX': 'LogError',
    
    }

autoSkimEXO = {
    'SingleMu' : 'WMu+ZMu+LogError+HWW+EXOLLRes',
    'DoubleMu' : 'WMu+ZMu+LogError+HWW+EXOLLRes',
    'DoubleElectron' : 'ZElectron+LogError+HWW+EXOLLRes',
    'MuEG' : 'LogError+HWW+EXOLLRes',
    }

autoSkimPDWG = {
    
    }

autoSkimDPG = {

    }

def mergeMapping(map1,map2):
    merged={}
    for k in list(set(map1.keys()+map2.keys())):
        items=[]
        if k in map1: 
            items.append(map1[k])
        if k in map2:
            items.append(map2[k])
        merged[k]='+'.join(items)
    return merged
    
#autoSkim = mergeMapping(autoSkimPDWG,autoSkimDPG)
#print autoSkim
