import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 50

### Configure alignment track selector
process.load( "Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi" )
process.AlignmentTrackSelector.src = cms.InputTag( "generalTracks" ) # for RelValZMM
#process.AlignmentTrackSelector.src = cms.InputTag( "ALCARECOTkAlZMuMu" ) # for ALCARECO
process.AlignmentTrackSelector.filter = cms.bool( True )
process.AlignmentTrackSelector.applyBasicCuts = cms.bool( True )
process.AlignmentTrackSelector.applyNHighestPt = cms.bool( True )
process.AlignmentTrackSelector.TwoBodyDecaySelector.applyMassrangeFilter = cms.bool( True )
process.AlignmentTrackSelector.TwoBodyDecaySelector.minXMass = cms.double( 80. )
process.AlignmentTrackSelector.TwoBodyDecaySelector.maxXMass = cms.double( 100. )

### Producer for TwoBodyDecay momentum constraint
process.load("RecoTracker.TrackProducer.TwoBodyDecayMomConstraintProducer_cff")
#process.TwoBodyDecayMomConstraint.chi2Cut = 100

### Producer for full TwoBodyDecay constraint
process.load("RecoTracker.TrackProducer.TwoBodyDecayConstraintProducer_cff")
#process.TwoBodyDecayConstraint.chi2Cut = 100

### KFFittingSmoother without outlier rejection - to be used for constrained fit
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import KFFittingSmoother
process.TwoBodyDecayTrackFitter = KFFittingSmoother.clone(
    ComponentName = cms.string( "TwoBodyDecayTrackFitter" ),
    Fitter = cms.string('RKFitter'),
    Smoother = cms.string('RKSmoother'),
    LogPixelProbabilityCut = cms.double(-15.0),
    EstimateCut = cms.double(-1.0),
)

from RecoTracker.TrackProducer.TrackRefitter_cfi import TrackRefitter

### First refitter - constructs transient tracks and trajectories from persistent tracks
process.Refitter = TrackRefitter.clone(
    src = "AlignmentTrackSelector",
    TrajectoryInEvent = True
)

### Refitter using the TwoBodyDecay momentum constraint
process.TrackRefitterTBDMomConstraint = TrackRefitter.clone(
    src = "AlignmentTrackSelector",
    srcConstr = "TwoBodyDecayMomConstraint",
    Fitter = cms.string('TwoBodyDecayTrackFitter'),
    constraint = "momentum",
    TrajectoryInEvent = True
)

### Refitter using the full TwoBodyDecay constraint
process.TrackRefitterTBDConstraint = TrackRefitter.clone(
    src = "AlignmentTrackSelector",
    srcConstr = "TwoBodyDecayConstraint",
    Fitter = cms.string('TwoBodyDecayTrackFitter'),
    constraint = "trackParameters",
    TrajectoryInEvent = True
)

process.p = cms.Path( process.AlignmentTrackSelector * process.Refitter * 
                      process.TwoBodyDecayConstraint * process.TrackRefitterTBDConstraint *
                      process.TwoBodyDecayMomConstraint * process.TrackRefitterTBDMomConstraint )

process.load( "RecoTracker.Configuration.RecoTracker_cff" )
process.load( "Configuration.StandardSequences.Services_cff" )
process.load( "Configuration.StandardSequences.GeometryPilot2_cff" )
process.load( "Configuration.StandardSequences.MagneticField_38T_cff" )
process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = "START44_V1::All"

# process.GlobalTag.toGet = cms.VPSet(
#     cms.PSet( record = cms.string( "TrackerAlignmentRcd" ),
#               tag = cms.string( "TrackerIdealGeometry210_mc" ),
#               connect = cms.untracked.string( "frontier://FrontierProd/CMS_COND_31X_FROM21X" ) ),
#     cms.PSet( record = cms.string( "TrackerAlignmentErrorRcd" ),
#               tag = cms.string( "TrackerIdealGeometryErrors210_mc" ),
#               connect = cms.untracked.string( "frontier://FrontierProd/CMS_COND_31X_FROM21X" ) )
# )

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        #"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_4_0_pre3/RelValZMM/GEN-SIM-RECO/START43_V4-v1/0001/805E5772-15A6-E011-9596-002618943800.root",
        "rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_4_0_pre3/RelValZMM/GEN-SIM-RECO/START43_V4-v1/0000/E6064D98-01A4-E011-AAA0-002618943956.root",
        #"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_4_0_pre3/RelValZMM/GEN-SIM-RECO/START43_V4-v1/0000/7C358E88-10A4-E011-BCDA-00304866C398.root",
        #"rfio:/castor/cern.ch/cms/store/relval/CMSSW_4_4_0_pre3/RelValZMM/GEN-SIM-RECO/START43_V4-v1/0000/5AF156B6-FDA3-E011-BDB7-0026189438C1.root"
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "output.root" ),
    outputCommands = cms.untracked.vstring("keep *Track*_*_*_*")
)
#process.outpath = cms.EndPath(process.out)
