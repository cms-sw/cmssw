import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cff import *
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cfi import *
from RecoParticleFlow.PFTracking.mergedElectronSeeds_cfi import *

electronSeeds = cms.Sequence(trackerDrivenElectronSeeds*ecalDrivenElectronSeeds*electronMergedSeeds) 
_electronSeedsFromMC = electronSeeds.copy()
_electronSeedsFromMC += cms.Sequence(ecalDrivenElectronSeedsFromMC*electronMergedSeedsFromMC)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
  electronSeeds, _electronSeedsFromMC
)


from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import *
electronGsfTracking = cms.Sequence(electronSeeds*electronCkfTrackCandidates*electronGsfTracks)
_electronGsfTracking = electronGsfTracking.copy()
_electronGsfTracking += cms.Sequence(electronCkfTrackCandidatesFromMC*electronGsfTracksFromMC)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
  electronGsfTracking, _electronGsfTracking
)
