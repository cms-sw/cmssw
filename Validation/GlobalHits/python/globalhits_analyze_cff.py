# The following comments couldn't be translated into the new config version:

# needed geometries
#
# needed backend

import FWCore.ParameterSet.Config as cms

from DQMServices.Components.test.dqm_onlineEnv_cfi import *
# actual producer
from Validation.GlobalHits.globalhits_analyze_cfi import *
DQMStore = cms.Service("DQMStore")

dqmSaver.convention = 'RelVal'
dqmSaver.workflow = '/GlobalValidation/Test/RECO'

