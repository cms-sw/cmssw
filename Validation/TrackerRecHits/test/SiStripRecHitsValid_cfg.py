import FWCore.ParameterSet.Config as cms

process = cms.Process("siStripRecHitsValid")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Validation.TrackerRecHits.SiStripRecHitsValid_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('/store/relval/2008/6/22/RelVal-RelValSingleMuPt10-1214048167-IDEAL_V2-2nd/0004/0AE2B3E3-0141-DD11-846F-000423D98BC4.root')
)

process.p1 = cms.Path(process.mix*process.siStripMatchedRecHits*process.stripRecHitsValid)


