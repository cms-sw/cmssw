import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
# Pattern Recognition and Fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
#FILTER
nuclearCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone()
import RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi
#TRAJECTORY BUILDER
nuclearCkfTrajectoryBuilder = RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi.CkfTrajectoryBuilder.clone()
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
nuclearCkfTrajectoryFilter.minPt = 0.3
nuclearCkfTrajectoryFilter.maxLostHits = 1
nuclearCkfTrajectoryFilter.minimumNumberOfHits = 3
nuclearCkfTrajectoryBuilder.trajectoryFilter.refToPSet_ = 'nuclearCkfTrajectoryFilter'
nuclearCkfTrajectoryBuilder.alwaysUseInvalidHits = False
nuclearTrackCandidates.src = 'nuclearSeed'
nuclearTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'nuclearCkfTrajectoryBuilder'
nuclearTrackCandidates.RedundantSeedCleaner = 'none'
nuclearWithMaterialTracks.src = 'nuclearTrackCandidates'
#FittingSmootherRK.MinNumberOfHits = 3

