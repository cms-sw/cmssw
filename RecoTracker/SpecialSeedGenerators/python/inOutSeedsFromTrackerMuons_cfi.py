import FWCore.ParameterSet.Config as cms

inOutSeedsFromTrackerMuons = cms.EDProducer("MuonReSeeder",
    ## Input collection of muons, and selection. track.isNonnull is implicit.
    src = cms.InputTag("muons"),
    cut = cms.string("pt > 2"),
    ## Keep at most these layers from the tracker track of the muon
    layersToKeep = cms.int32(5),
    ## Use the inner part of the tracker track to make a seed
    insideOut  = cms.bool(True),
    #### Turn on verbose debugging (to be removed at the end)
    debug = cms.untracked.bool(False),
    ## Configuration for the refitter
    DoPredictionsOnly = cms.bool(False),
    Fitter = cms.string('KFFitterForRefitInsideOut'),
    TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
    Smoother = cms.string('KFSmootherForRefitInsideOut'),
    MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
    MTDRecHitBuilder = cms.string('MTDRecHitBuilder'),
    RefitDirection = cms.string('alongMomentum'),
    RefitRPCHits = cms.bool(True),
    Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
)
