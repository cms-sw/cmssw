import FWCore.ParameterSet.Config as cms

process = cms.Process("processPixelRecHitValid")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_30X::All'


process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Validation.TrackerRecHits.SiPixelRecHitsValid_cfi")
process.pixRecHitsValid.outputFile="pixelrechitshisto.root"

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RECO/IDEAL_30X_v3/0005/1EF32A82-57E2-DD11-A475-000423D6B444.root')
)

process.p1 = cms.Path(process.mix*process.siPixelRecHits*process.pixRecHitsValid)


