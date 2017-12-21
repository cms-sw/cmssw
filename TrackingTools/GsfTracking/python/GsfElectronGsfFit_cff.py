import FWCore.ParameterSet.Config as cms

# Gsf track fit for GsfElectrons
from TrackingTools.GsfTracking.fwdGsfElectronPropagator_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
import TrackingTools.GsfTracking.GsfElectronFit_cfi
electronGsfTracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
electronGsfTracks.src = 'electronCkfTrackCandidates'
electronGsfTracks.Propagator = 'fwdGsfElectronPropagator'
electronGsfTracks.Fitter = 'GsfElectronFittingSmoother'
electronGsfTracks.TTRHBuilder = 'WithTrackAngle'
electronGsfTracks.TrajectoryInEvent = False

# FastSim has no template fit on tracker hits
# replace the ECAL driven electron track candidates with the FastSim emulated ones
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(electronGsfTracks,
                 src = "fastElectronCkfTrackCandidates",
                 TTRHBuilder = "WithoutRefit")

electronGsfTracksFromMultiCl = electronGsfTracks.clone(
  src = 'electronCkfTrackCandidatesFromMultiCl'
)

