import FWCore.ParameterSet.Config as cms

# Gsf track fit for GsfElectrons
from TrackingTools.GsfTracking.fwdGsfElectronPropagator_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
import TrackingTools.GsfTracking.GsfElectronFit_cfi
electronGsfTracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone(
    src = 'electronCkfTrackCandidates',
    Propagator = 'fwdGsfElectronPropagator',
    Fitter = 'GsfElectronFittingSmoother',
    TTRHBuilder = 'WithTrackAngle',
    TrajectoryInEvent = False
)
# FastSim has no template fit on tracker hits
# replace the ECAL driven electron track candidates with the FastSim emulated ones
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(electronGsfTracks,
                 src = "fastElectronCkfTrackCandidates",
                 TTRHBuilder = "WithoutRefit")

electronGsfTracksFromMultiCl = electronGsfTracks.clone(
  src = 'electronCkfTrackCandidatesFromMultiCl'
)
