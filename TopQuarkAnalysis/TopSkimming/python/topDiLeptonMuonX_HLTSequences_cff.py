import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topDiLeptonMuonXHLT = copy.deepcopy(hltHighLevel)
topDiLeptonMuonXHLT.HLTPaths = ['HLT_Mu15_L1Mu7', 'HLT_DoubleMu3', 'HLT_IsoEle10_Mu10_L1R']

