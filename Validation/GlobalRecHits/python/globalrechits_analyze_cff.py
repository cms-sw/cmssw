# The following comments couldn't be translated into the new config version:

# needed backend

import FWCore.ParameterSet.Config as cms

from DQMServices.Components.test.dqm_onlineEnv_cfi import *
# actual producer
from Validation.GlobalRecHits.globalrechits_analyze_cfi import *
DQMStore = cms.Service("DQMStore")

dqmSaver.fileName = 'GlobalRecHitsAnalyze'
dqmSaver.saveAtRunEnd = False
dqmSaver.saveAtJobEnd = True

