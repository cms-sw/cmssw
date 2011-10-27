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

    'HT': 'HighMET+EXOHT+LogError',

    'Tau': 'LogError',
    'PhotonHad': 'LogError',
    'MuHad': 'LogError',
    'MultiJet': 'LogError',
    'MuOnia': 'LogError',
    'ElectronHad': 'LogError',
    'TauPlusX': 'LogError',
    
    }

## autoSkimEXO = {
##     'BTag' : 'EXOHSCP+LogError',
##     'DoubleElectron' : 'EXOHPTE+EXOTriEle+EXODiEle+HWW+HZZ+LogError',
##     'DoubleMu' : 'EXOLongLivedMu+EXOHSCP+EXOTriMu+EXODiMu+HWW+HZZ+LogError',
##     'HT': 'EXOHT+LogError',
##     'MET': 'EXOHSCP+EXOSingleJet+LogError',
##     'METBTag': 'EXOHSCP+EXOSingleJet+LogError',
##     'MuEG' : 'EXO1E2Mu+HWW+HZZ+LogError',
##     'Photon': 'EXOLongLivedPhoton+DoublePhoton+LogError',
##     'SingleElectron' : 'EXOEle+EXOHPTE+HWW+HZZ+LogError',
##     'SingleMu' : 'EXOHSCP+EXOMu+HWW+HZZ+LogError',
##     }


autoSkimEXO = {
##    'BTag' : 'EXOHSCP+LogError',
    'DoubleElectron' : 'EXOTriEle+EXODiEle+HWW+HZZ+LogError',
    'DoubleMu' : 'EXOTriMu+EXODiMu+HWW+HZZ+LogError',
    'HT': 'EXOHT+LogError',
    'MET': 'EXOSingleJet+LogError',
    'METBTag': 'EXOSingleJet+LogError',
    'MuEG' : 'EXO1E2Mu+HWW+HZZ+LogError',
    'Photon': 'DoublePhoton+EXOHPTE+LogError',
    'SingleElectron' : 'EXOEle+HWW+HZZ+LogError',
    'SingleMu' : 'EXOMu+HWW+HZZ+LogError',
    }

## autoSkimEXO = {
##     'EXOTriEle' : 'EXOTriEle',
##     'EXODiEle' : 'EXODiEle',
##     'EXOLongLivedMu' : 'EXOLongLivedMu',
##     'EXOTriMu' : 'EXOTriMu',
##     'EXODiMu' : 'EXODiMu',
##     'EXOHT' : 'EXOHT',
##     'EXOSingleJet' : 'EXOSingleJet',
##     'EXO1E2Mu' : 'EXO1E2Mu',
##     'EXOLongLivedPhoton' : 'EXOLongLivedPhoton',
##     'EXOEle' : 'EXOEle',
##     'EXOMu' : 'EXOMu',
##     'HWW' : 'HWW',
##     'HZZ' : 'HZZ',
##     'EXOHSCP' : 'EXOHSCP',
##     'DoublePhoton' : 'DoublePhoton',
##     'EXOHPTE' : 'EXOHPTE',
##     }

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
