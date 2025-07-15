import FWCore.ParameterSet.Config as cms

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi 
hitCollectorForOutInMuonSeeds = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'hitCollectorForOutInMuonSeeds',
    MaxChi2       = 100.0, ## was 30 ## TO BE TUNED
    nSigma        = 4.,    ## was 3  ## TO BE TUNED 
)

from RecoTracker.SpecialSeedGenerators.outsideInMuonSeeder_cfi import outsideInMuonSeeder
outInSeedsFromStandaloneMuons = outsideInMuonSeeder.clone(
    ## Input collection of muons, and selection. outerTrack.isNonnull is implicit.
    src = 'muons',
    cut = 'pt > 10 && outerTrack.hitPattern.muonStationsWithValidHits >= 2',
    layersToTry = 3, # try up to 3 layers where at least one seed is found
    hitsToTry = 3,   # use at most 3 hits from the same layer
    ## Use as state the muon updated ad vertex (True) or the innermost state of the standalone track (False)
    fromVertex = True,
    ## Propagator to go from muon state to TOB/TEC.
    muonPropagator = 'SteppingHelixPropagatorAlong',
    ## Propagator used searching for hits..
    trackerPropagator  = 'PropagatorWithMaterial',
    ## How much to rescale the standalone muon uncertainties beforehand
    errorRescaleFactor = 2.0,
    ## Chi2MeasurementEstimator used to select hits
    hitCollector = 'hitCollectorForOutInMuonSeeds',
    ## Eta ranges to search for TOB and TEC
    maxEtaForTOB = 1.8,
    minEtaForTEC = 0.7,
    #### Turn on verbose debugging (to be removed at the end)
    debug = False
)

