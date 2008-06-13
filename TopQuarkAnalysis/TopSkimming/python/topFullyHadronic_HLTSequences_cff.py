import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
topFullyHadronicJetsHLT = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# added all b-jet HLT paths for testing purposes
topFullyHadronicBJetsHLT = copy.deepcopy(hltHighLevel)
topFullyHadronicJetsHLT.HLTPaths = ['HLT1jet', 'HLT_DoubleJet150', 'HLT_TripleJet85', 'HLT_QuadJet60']
topFullyHadronicBJetsHLT.HLTPaths = ['HLT_BTagIP_Jet180', 'HLT_BTagIP_DoubleJet120', 'HLT_BTagIP_TripleJet70', 'HLT_BTagIP_QuadJet40']

