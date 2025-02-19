import FWCore.ParameterSet.Config as cms

from TrackingTools.GsfTracking.CkfElectronCandidates_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
GsfGlobalElectronTestSequence = cms.Sequence(CkfElectronCandidates*GsfGlobalElectronTest)

