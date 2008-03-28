import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
#Magnetic Field and Geometry
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
# NavigationSchool
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
# Pattern Recognition and Fit
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
#FILTER
nuclearCkfTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
#TRAJECTORY BUILDER
nuclearCkfTrajectoryBuilder = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#TRACK CANDIDATES
nuclearTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#TRACKS
nuclearWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
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
nuclearTrackCandidates.SeedProducer = 'nuclearSeed'
nuclearTrackCandidates.TrajectoryBuilder = 'nuclearCkfTrajectoryBuilder'
nuclearTrackCandidates.RedundantSeedCleaner = 'none'
nuclearWithMaterialTracks.src = 'nuclearTrackCandidates'
FittingSmootherRK.MinNumberOfHits = 3

