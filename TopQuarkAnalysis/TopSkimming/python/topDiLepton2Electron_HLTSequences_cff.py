import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topDiLepton2ElectronHLT = copy.deepcopy(hltHighLevel)
topDiLepton2ElectronHLT.HLTPaths = ['HLT_IsoEle18_L1R', 'HLT_DoubleIsoEle12_L1R']

