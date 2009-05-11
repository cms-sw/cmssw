import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TtSemiLepKinFitter')
process.MessageLogger.categories.append('KinFitter')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default             = cms.untracked.PSet( limit = cms.untracked.int32( 0) ),
    TtSemiLepKinFitter  = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    KinFitter           = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

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

## std sequence to produce the kinematic fit for semi-leptonic events
process.load("TopQuarkAnalysis.TopKinFitter.TtSemiLepKinFitProducer_Muons_cfi")

## process path
process.p = cms.Path(process.patDefaultSequence *
                     process.kinFitTtSemiLepEvent
                     )

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents   = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),                               
    fileName = cms.untracked.string('ttSemiLepKinFitProducer.root'),
    outputCommands = cms.untracked.vstring('drop *')
)
process.out.outputCommands += ['keep *_kinFitTtSemiLepEvent_*_*']

## output path
process.outpath = cms.EndPath(process.out)
