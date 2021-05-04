import FWCore.ParameterSet.Config as cms

from TrackingTools.GsfTracking.CkfElectronCandidates_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
GsfGlobalElectronTestTask = cms.Task(CkfElectronCandidates,GsfGlobalElectronTest)
GsfGlobalElectronTestSequence = cms.Sequence(GsfGlobalElectronTestTask)
