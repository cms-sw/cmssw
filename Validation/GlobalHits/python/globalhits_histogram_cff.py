# The following comments couldn't be translated into the new config version:

# needed backend

import FWCore.ParameterSet.Config as cms

# actual producer
from Validation.GlobalHits.globalhits_histogram_cfi import *
DQMStore = cms.Service("DQMStore")


