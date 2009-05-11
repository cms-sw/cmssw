import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

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

## configure mva computer
process.load("TopQuarkAnalysis.TopJetCombination.TtSemiLepJetCombMVAComputer_cff")
## change maximum number of jets taken into account per event (default: 4)
#process.findTtSemiLepJetCombMVA.maxNJets = 5

## produce pat objects and perform MVA for jet-parton association
process.p = cms.Path(process.patDefaultSequence *
                     process.findTtSemiLepJetCombMVA)

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents   = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),                               
    fileName = cms.untracked.string('ttSemiLepJetCombMVAComputer_muons.root'),
    outputCommands = cms.untracked.vstring('drop *')
)
process.out.outputCommands += ['keep *_findTtSemiLepJetCombMVA_*_*']

## output path
process.outpath = cms.EndPath(process.out)
