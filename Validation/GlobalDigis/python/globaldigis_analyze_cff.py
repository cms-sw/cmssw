# The following comments couldn't be translated into the new config version:

# needed backend

import FWCore.ParameterSet.Config as cms

from DQMServices.Components.test.dqm_onlineEnv_cfi import *
# actual producer
from Validation.GlobalDigis.globaldigis_analyze_cfi import *
DQMStore = cms.Service("DQMStore")

dqmSaver.convention = 'RelVal'
dqmSaver.workflow = 'GlobalValidation/Test/RECO'

# foo bar baz
# 0n2zCI7PJQ3qy
