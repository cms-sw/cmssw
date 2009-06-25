import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.categories = cms.untracked.vstring('TEST')

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0002/50D4BADB-FA32-DE11-BA01-000423D98DC4.root'
     ),
     skipEvents = cms.untracked.uint32(0)                            
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry & conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('IDEAL_31X::All')

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

from TopQuarkAnalysis.Examples.TopTauAnalyzer_cfi import analyzeTau
process.analyzeTau = analyzeTau

# register TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzeTopTau.root')
)

## end path   
process.p1 = cms.Path(process.patDefaultSequence *
                      process.analyzeTau)

