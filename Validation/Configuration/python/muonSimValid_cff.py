import FWCore.ParameterSet.Config as cms

# muon validation sequences
from Validation.MuonHits.muonHitsValidation_cfi import *
from Validation.MuonDTDigis.dtDigiValidation_cfi import *
from Validation.MuonCSCDigis.cscDigiValidation_cfi import *
from Validation.MuonRPCDigis.validationMuonRPCDigis_cfi import *
from Validation.RecoMuon.muonValidation_cff import *

muonSimValid = cms.Sequence(validSimHit+muondtdigianalyzer+cscDigiValidation+validationMuonRPCDigis+recoMuonValidation)
