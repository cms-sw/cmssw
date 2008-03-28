import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topFullyHadronicJetsHLT = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# added all b-jet HLT paths for testing purposes
topFullyHadronicBJetsHLT = copy.deepcopy(hltHighLevel)
topFullyHadronicJetsHLT.HLTPaths = ['HLT1jet', 'HLT2jet', 'HLT3jet', 'HLT4jet']
topFullyHadronicBJetsHLT.HLTPaths = ['HLTB1Jet', 'HLTB2Jet', 'HLTB3Jet', 'HLTB4Jet']

