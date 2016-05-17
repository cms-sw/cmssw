import FWCore.ParameterSet.Config as cms

# seeding
from RecoMuon.CosmicMuonProducer.cosmicMuons_cfi import *
import RecoMuon.MuonIdentification.muons1stStep_cfi
muonsForCosmicDC = RecoMuon.MuonIdentification.muons1stStep_cfi.muons1stStep.clone(
    inputCollectionLabels = cms.VInputTag("cosmicMuons"),
    inputCollectionTypes = cms.vstring('outer tracks'),
    fillIsolation = cms.bool(False),
    fillGlobalTrackQuality = cms.bool(False),
    fillGlobalTrackRefits = cms.bool(False),
)
muonsForCosmicDC.TrackExtractorPSet.inputTrackCollection = cms.InputTag("cosmicMuons")
import RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi
import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi 
hitCollectorForOutInMuonSeeds = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('hitCollectorForOutInMuonSeeds'),
    MaxChi2 = cms.double(100.0), ## was 30 ## TO BE TUNED
    nSigma  = cms.double(4.),    ## was 3  ## TO BE TUNED 
    MaxDisplacement = cms.double(100),
    MaxSagitta = cms.double(-1.0),
    MinimalTolerance = cms.double(0.5),
    appendToDataLabel = cms.string(''),
)
cosmicDCSeeds = RecoTracker.SpecialSeedGenerators.outInSeedsFromStandaloneMuons_cfi.outInSeedsFromStandaloneMuons.clone(
    src = cms.InputTag("muonsForCosmicDC"),
    cut = cms.string("pt > 2 && abs(eta)<1.2 && phi<0"),
    fromVertex = cms.bool(False),
    maxEtaForTOB = cms.double(1.5),
    minEtaForTEC = cms.double(0.7),
)

# Ckf pattern
import RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff
cosmicDCCkfTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidatesP5_cff.ckfTrackCandidatesP5.clone(
    src = cms.InputTag( "cosmicDCSeeds" ),
)

# Track producer
import RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff
cosmicDCTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterialP5_cff.ctfWithMaterialTracksCosmics.clone(
    src = cms.InputTag( "cosmicDCCkfTrackCandidates" ),
)

# Final Sequence
cosmicDCTracksSeq = cms.Sequence( cosmicMuons * muonsForCosmicDC * cosmicDCSeeds * cosmicDCCkfTrackCandidates * cosmicDCTracks )
