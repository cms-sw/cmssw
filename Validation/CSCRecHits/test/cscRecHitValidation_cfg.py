import FWCore.ParameterSet.Config as cms

process = cms.Process("RecHitsValidation")
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

process.load("Validation.CSCRecHits.cscRecHitValidation_cfi")

process.load("Validation.MuonCSCDigis.cscDigiValidation_cfi")

process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:simevent.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.p1 = cms.Path(process.mix*process.simMuonCSCDigis*process.cscDigiValidation*process.csc2DRecHits*process.cscSegments*process.cscRecHitValidation)
process.cscRecHitValidation.outputFile = 'cscRecHitValidation_ref.root'


