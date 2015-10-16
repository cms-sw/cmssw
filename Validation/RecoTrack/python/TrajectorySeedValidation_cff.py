import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.trajectorySeedTracks_cfi import trajectorySeedTracks as _trajectorySeedTracks
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import trackingParticleRecoTrackAsssociation as _trackAssociation
from Validation.RecoTrack.TrackValidation_cff import *

_seedProducerLabels = ["initialStepSeeds",
                       "detachedTripletStepSeeds",
                       "lowPtTripletStepSeeds",
                       "pixelPairStepSeeds",
                       "mixedTripletStepSeedsA",
                       "mixedTripletStepSeedsB",
                       "pixelLessStepSeeds",
                       "tobTecStepSeedsPair",
                       "tobTecStepSeedsTripl",
                       "jetCoreRegionalStepSeeds",
                       ]

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    _seedProducerLabels.remove("jetCoreRegionalStepSeeds")

_moduleNames = []
for _label in _seedProducerLabels:
    _lines = """
{0}Tracks = _trajectorySeedTracks.clone(src = cms.InputTag(\"{0}\"))
_moduleNames.extend([\"{0}Tracks\"])
""".format(_label)
    exec(_lines)


trajectorySeedValidator = trackValidator.clone(
    dodEdxPlots = False,
    label = [cms.InputTag(x) for x in _moduleNames],
    UseAssociators=True, 
    associators=[cms.InputTag("quickTrackAssociatorByHits")]
    )

_line = "trajectorySeedValidation = cms.Sequence(quickTrackAssociatorByHits+{0}+trajectorySeedValidator)".format("+".join(_moduleNames))
exec(_line)


tracksAndTrajectorySeedsValidationStandalone = cms.Sequence(
    tracksValidationStandalone +
    trajectorySeedValidation
)


# 'slim' sequences that only depend on track, seed, and tracking particle collections
trajectorySeedValidatorSlim = trajectorySeedValidator.clone(
    doPVAssociationPlots = cms.untracked.bool(False),
)
trajectorySeedValidationSlim = trajectorySeedValidation.copy()
trajectorySeedValidationSlim.replace(trajectorySeedValidator,trajectorySeedValidatorSlim)

tracksAndTrajectorySeedsValidationSlim = cms.Sequence(
    tracksValidationSlim +
    trajectorySeedValidationSlim
)
