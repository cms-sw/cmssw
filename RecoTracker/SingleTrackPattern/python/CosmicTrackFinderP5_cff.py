import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.SingleTrackPattern.CosmicCandidateFinder_cfi import *
#cosmictrackfinderP5 = copy.deepcopy(cosmictrackfinder)
#TEMPORARY FIX: actually it is not yet a candidateMaker
cosmicCandidateFinderP5 = copy.deepcopy(cosmicCandidateFinder)
cosmicCandidateFinderP5.GeometricStructure = 'STANDARD'
cosmicCandidateFinderP5.cosmicSeeds = 'cosmicseedfinderP5'

from RecoTracker.TrackProducer.CosmicFinalFitWithMaterialP5_cff import *
#import RecoTracker.TrackProducer.TrackRefitter_cfi 
#cosmictrackfinderP5  = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(
#    src = cms.InputTag("cosmicCandidateFinderP5"),
#    Fitter = cms.string('FittingSmootherRKP5'),
#    TTRHBuilder = cms.string('WithTrackAngle'),
#    AlgorithmName = cms.string('cosmics')
#)
