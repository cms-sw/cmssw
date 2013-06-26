# The following comments couldn't be translated into the new config version:

# needed backend

import FWCore.ParameterSet.Config as cms

# actual producer
from Validation.GlobalRecHits.globalrechitshistogram_cfi import *
DQMStore = cms.Service("DQMStore")


