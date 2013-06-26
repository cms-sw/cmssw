import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
# Pattern Recognition and Fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
#FILTER
nuclearCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi
#TRAJECTORY BUILDER
nuclearCkfTrajectoryBuilder = RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi.CkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
#TRACK CANDIDATES
nuclearTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
#TRACKS
nuclearWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
from RecoTracker.NuclearSeedGenerator.NuclearSeed_cfi import *
from RecoVertex.NuclearInteractionProducer.NuclearInteraction_cfi import *
from RecoTracker.NuclearSeedGenerator.NuclearTrackCorrector_cfi import *
nuclear_interaction = cms.Sequence(nuclearSeed*nuclearTrackCandidates*nuclearWithMaterialTracks*nuclearInteractionMaker)
nuclear_interaction_and_correction = cms.Sequence(nuclear_interaction*TrackCorrector)
nuclearCkfTrajectoryFilter.ComponentName = 'nuclearCkfTrajectoryFilter'
nuclearCkfTrajectoryFilter.filterPset.minPt = 0.3
nuclearCkfTrajectoryFilter.filterPset.maxLostHits = 1
nuclearCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
nuclearCkfTrajectoryBuilder.ComponentName = 'nuclearCkfTrajectoryBuilder'
nuclearCkfTrajectoryBuilder.trajectoryFilterName = 'nuclearCkfTrajectoryFilter'
nuclearCkfTrajectoryBuilder.alwaysUseInvalidHits = False
nuclearTrackCandidates.src = 'nuclearSeed'
nuclearTrackCandidates.TrajectoryBuilder = 'nuclearCkfTrajectoryBuilder'
nuclearTrackCandidates.RedundantSeedCleaner = 'none'
nuclearWithMaterialTracks.src = 'nuclearTrackCandidates'
#FittingSmootherRK.MinNumberOfHits = 3

