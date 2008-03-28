import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topDiLeptonMuonXHLT = copy.deepcopy(hltHighLevel)
topDiLeptonMuonXHLT.HLTPaths = ['HLT1MuonNonIso', 'HLT2MuonNonIso', 'HLTXElectronMuonRelaxed']

