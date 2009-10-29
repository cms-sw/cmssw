import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerValidation")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V5::All'


process.load("Configuration.StandardSequences.Services_cff")

process.load("SimG4Core.Configuration.SimG4Core_cff")

process.load("Validation.TrackerHits.trackerHitsValidation_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(#'/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/38E34C97-E8DD-DD11-8327-000423D94534.root')
'/store/relval/CMSSW_3_2_5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V5-v1/0011/F2673349-5B8E-DE11-93F3-000423D99EEE.root',
'/store/relval/CMSSW_3_2_5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V5-v1/0011/D6143575-5B8E-DE11-BF77-001D09F2AF1E.root',
'/store/relval/CMSSW_3_2_5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V5-v1/0011/B832B99D-598E-DE11-AFF7-000423D6B48C.root',
'/store/relval/CMSSW_3_2_5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V5-v1/0011/8C9CAA3D-538E-DE11-ABDC-001617E30F4C.root',
'/store/relval/CMSSW_3_2_5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V5-v1/0011/827F235F-588E-DE11-9A2F-003048D2C0F0.root',
'/store/relval/CMSSW_3_2_5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V5-v1/0011/628072A8-578E-DE11-8A36-003048D3750A.root',
'/store/relval/CMSSW_3_2_5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V5-v1/0011/24B1F50B-838E-DE11-AA5A-000423D98920.root',
'/store/relval/CMSSW_3_2_5/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V5-v1/0011/1C5666DB-5A8E-DE11-A1E1-000423D94C68.root'

    )
)

#process.o1 = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('Muon_FullValidation.root')
#)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.trackerHitsValid.outputFile ="TrackerHitHisto.root"
process.p1 = cms.Path(process.g4SimHits*process.trackerHitsValidation)
#process.outpath = cms.EndPath(process.o1)


