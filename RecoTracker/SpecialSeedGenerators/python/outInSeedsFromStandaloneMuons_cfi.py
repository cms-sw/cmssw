import FWCore.ParameterSet.Config as cms

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi 
hitCollectorForOutInMuonSeeds = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('hitCollectorForOutInMuonSeeds'),
    MaxChi2 = cms.double(100.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(4.),    ## was 3  ## TO BE TUNED 
)

outInSeedsFromStandaloneMuons = cms.EDProducer("OutsideInMuonSeeder",
    ## Input collection of muons, and selection. outerTrack.isNonnull is implicit.
    src = cms.InputTag("muons"),
    cut = cms.string("pt > 10 && outerTrack.hitPattern.muonStationsWithValidHits >= 2"),
    layersToTry = cms.int32(3), # try up to 3 layers where at least one seed is found
    hitsToTry = cms.int32(3),   # use at most 3 hits from the same layer
    ## Use as state the muon updated ad vertex (True) or the innermost state of the standalone track (False)
    fromVertex = cms.bool(True),
    ## Propagator to go from muon state to TOB/TEC.
    muonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    ## Propagator used searching for hits..
    trackerPropagator  = cms.string('PropagatorWithMaterial'),
    ## How much to rescale the standalone muon uncertainties beforehand
    errorRescaleFactor = cms.double(2.0),
    ## Chi2MeasurementEstimator used to select hits
    hitCollector = cms.string('hitCollectorForOutInMuonSeeds'),
    ## Eta ranges to search for TOB and TEC
    maxEtaForTOB = cms.double(1.8),
    minEtaForTEC = cms.double(0.7),
    #### Turn on verbose debugging (to be removed at the end)
    debug = cms.untracked.bool(False),
)

