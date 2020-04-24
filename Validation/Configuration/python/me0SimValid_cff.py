import FWCore.ParameterSet.Config as cms

from Validation.MuonME0Validation.MuonME0Hits_cfi import *
from Validation.MuonME0Validation.MuonME0Digis_cfi import *
from Validation.MuonME0Validation.MuonME0RecHits_cfi import *

me0SimValid = cms.Sequence(me0HitsValidation*me0DigiValidation*me0LocalRecoValidation)
