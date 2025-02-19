# The following comments couldn't be translated into the new config version:

# SEEDS

import FWCore.ParameterSet.Config as cms

from RecoVertex.NuclearInteractionProducer.NuclearInteraction_cff import *
import copy
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
#TRACKER HITS
nuclearPixelRecHits = copy.deepcopy(siPixelRecHits)
import copy
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
nuclearStripRecHits = copy.deepcopy(siStripMatchedRecHits)
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#TRAJECTORY MEASUREMENT
nuclearMeasurementTracker = copy.deepcopy(MeasurementTracker)
#HIT REMOVAL
nuclearClusters = cms.EDProducer("RemainingClusterProducer",
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    recTracks = cms.InputTag("ctfWithMaterialTracks"),
    DistanceCut = cms.double(0.1),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    pixelHits = cms.InputTag("siPixelRecHits")
)

nuclear_seed_second = cms.Sequence(nuclearClusters*nuclearPixelRecHits*nuclearStripRecHits*nuclearSeed)
nuclear_interaction_second = cms.Sequence(nuclear_seed_second*nuclearTrackCandidates*nuclearWithMaterialTracks*nuclearInteractionMaker)
nuclear_interaction_second_and_correction = cms.Sequence(nuclear_interaction_second*TrackCorrector)
nuclearPixelRecHits.src = 'nuclearClusters'
nuclearStripRecHits.ClusterProducer = 'nuclearClusters'
nuclearMeasurementTracker.ComponentName = 'nuclearMeasurementTracker'
nuclearMeasurementTracker.pixelClusterProducer = 'nuclearClusters'
nuclearMeasurementTracker.stripClusterProducer = 'nuclearClusters'
nuclearSeed.MeasurementTrackerName = 'nuclearMeasurementTracker'
#TRAJECTORY BUILDER
nuclearCkfTrajectoryBuilder.MeasurementTrackerName = 'nuclearMeasurementTracker'

