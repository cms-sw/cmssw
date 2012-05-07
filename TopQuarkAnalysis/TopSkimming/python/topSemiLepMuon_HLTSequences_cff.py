import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topSemiLepMuonHLT = copy.deepcopy(hltHighLevel)
topSemiLepMuonHLT.HLTPaths = ['HLT_Mu15_L1Mu7']

