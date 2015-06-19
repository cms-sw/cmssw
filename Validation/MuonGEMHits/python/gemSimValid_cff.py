#### This file must be moved to "Validation/Configuration/"
import FWCore.ParameterSet.Config as cms

from Validation.MuonGEMHits.MuonGEMHits_cfi import *
from Validation.MuonGEMDigis.MuonGEMDigis_cfi import *
#from Validation.MuonGEMRecHits.MuonGEMRecHits_cfi import *

#from Validation.MuonME0Hits.MuonME0Hits_cfi import *
#from Validation.MuonME0Digis.MuonME0Digis_cfi import *
#from Validation.MuonME0RecHits.MuonME0RecHits_cfi import *

gemSimValid = cms.Sequence(gemSimValidation*gemDigiValidation)
#gemSimValid = cms.Sequence(gemHitsValidation*gemDigiValidation*gemRecHitsValidation)
#me0SimValid = cms.Sequence(me0HitsValidation*me0DigiValidation*me0RecHitsValidation)
