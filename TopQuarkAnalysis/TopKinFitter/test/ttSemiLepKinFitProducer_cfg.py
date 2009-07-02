import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# test cfg file for the ttSemiLepKinFitProducer
#-------------------------------------------------
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

#-------------------------------------------------
# process configuration
#-------------------------------------------------


## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_1_X_2008-07-08_STARTUP_V4-AODSIM.100.root')
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")

## configure conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V4::All')

## load magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

#-------------------------------------------------
# tqaf configuration
#-------------------------------------------------

## std sequence for tqaf layer1
process.load("TopQuarkAnalysis.TopObjectProducers.tqafLayer1_cff")

## necessary fixes to run 2.2.X on 2.1.X data
## comment this when running on samples produced
## with 22X
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run22XonSummer08AODSIM
run22XonSummer08AODSIM(process)

## std sequence to produce the kinematic fit for semi-leptonic events
process.load("TopQuarkAnalysis.TopKinFitter.TtSemiLepKinFitProducer_Muons_cfi")

## process path
process.p = cms.Path(process.tqafLayer1  *
                     process.kinFitTtSemiLepEvent
                     )
