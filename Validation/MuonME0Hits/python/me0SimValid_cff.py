#### This file must be moved to "Validation/Configuration/"
import FWCore.ParameterSet.Config as cms

from Validation.MuonME0Hits.MuonME0Hits_cfi import *
#from Validation.MuonME0Digis.MuonME0Digis_cfi import *
#from Validation.MuonME0RecHits.MuonME0RecHits_cfi import *

me0SimValid = cms.Sequence(me0HitsValidation)
#me0SimValid = cms.Sequence(me0HitsValidation*me0DigiValidation*me0RecHitsValidation)
