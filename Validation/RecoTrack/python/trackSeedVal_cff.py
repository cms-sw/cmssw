import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.seedTracks_cfi import seedTracks as _seedTracks
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import trackingParticleRecoTrackAsssociation as _trackAssociation
from Validation.RecoTrack.TrackValidation_fastsim_cff import  quickTrackAssociatorByHits
from Validation.RecoTrack.TrackValidation_fastsim_cff import  trackValidator as _trackValidator

_seedProducerLabels = ["initialStepSeeds",
                       "detachedTripletStepSeeds",
                       #"jetCoreRegionalStepSeeds",
                       "lowPtTripletStepSeeds",
                       "pixelPairStepSeeds",
                       "mixedTripletStepSeedsA",
                       "mixedTripletStepSeedsB",
                       "pixelLessStepSeeds",
                       "tobTecStepSeedsPair",
                       "tobTecStepSeedsTripl"]

_moduleNames = []
for _label in _seedProducerLabels:
    _lines = """
{0}Tracks = _seedTracks.clone(src = cms.InputTag(\"{0}\"))
{0}Association = _trackAssociation.clone(label_tr = cms.InputTag(\"{0}Tracks\"))
{0}Validator = _trackValidator.clone(
   trackCollectionForDrCalculation = cms.InputTag("{0}Tracks"),
   label = [cms.InputTag("{0}Tracks")],
   associators = [cms.InputTag(\"{0}Association\")])

_moduleNames.extend([\"{0}Tracks\",\"{0}Association\",\"{0}Validator\"])
""".format(_label)
    exec(_lines)

_line = "seedValidationSequence = cms.Sequence(quickTrackAssociatorByHits+{0})".format("+".join(_moduleNames))
exec(_line)
