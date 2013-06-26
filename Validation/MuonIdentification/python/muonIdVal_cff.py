import FWCore.ParameterSet.Config as cms

from Validation.MuonIdentification.muonIdVal_cfi import *
from DQMOffline.Muon.muonIdDQM_cfi import *
muonIdDQMInVal = muonIdDQM.clone()
muonIdDQMInVal.useGlobalMuons = cms.untracked.bool(True)
muonIdDQMInVal.useGlobalMuonsNotTrackerMuons = cms.untracked.bool(True)
muonIdDQMInVal.baseFolder = muonIdVal.baseFolder

# MuonIdVal
muonIdValSeq = cms.Sequence(muonIdVal)
# MuonIdVal and MuonIdDQM
muonIdValDQMSeq = cms.Sequence(muonIdVal*muonIdDQMInVal)
