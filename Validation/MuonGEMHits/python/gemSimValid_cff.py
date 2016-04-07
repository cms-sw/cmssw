#### This file must be moved to "Validation/Configuration/"
import FWCore.ParameterSet.Config as cms

from Validation.MuonGEMHits.MuonGEMHits_cff import *
from Validation.MuonGEMDigis.MuonGEMDigis_cff import *
from Validation.MuonGEMRecHits.MuonGEMRecHits_cff import *


try :
  from Validation.MuonME0Hits.MuonME0Hits_cfi import *
  from Validation.MuonME0Digis.MuonME0Digis_cfi import *
  from Validation.MuonME0RecHits.MuonME0RecHits_cfi import *
except :
  print "Import error : Skip MuonME0 "
else :
  me0SimValid = cms.Sequence(me0HitsValidation*me0DigiValidation*me0RecHitsValidation)

#gemSimValid = cms.Sequence(gemSimValidation*gemDigiValidation)
gemSimValid = cms.Sequence(gemSimValidation*gemDigiValidation*gemRecHitsValidation)
