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
electronGsfTracks.TrajectoryInEvent = True

