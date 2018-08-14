import FWCore.ParameterSet.Config as cms

SimMuFilter = cms.EDFilter('SimMuFilter',

              simTracksInput = cms.InputTag("g4SimHits","","SIM"),
              simHitsMuonRPCInput = cms.InputTag("g4SimHits","MuonRPCHits","SIM"),
              simHitsMuonCSCInput = cms.InputTag("g4SimHits","MuonCSCHits","SIM"),
              simHitsMuonDTInput = cms.InputTag("g4SimHits","MuonDTHits","SIM"),
              nMuSel = cms.int32(1)
)
