import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topDiLepton2ElectronHLT = copy.deepcopy(hltHighLevel)
topDiLepton2ElectronHLT.HLTPaths = ['HLT1ElectronRelaxed', 'HLT2ElectronRelaxed']

