import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topSemiLepElectronHLT = copy.deepcopy(hltHighLevel)
topSemiLepElectronHLT.HLTPaths = ['HLT_IsoEle18_L1R']

