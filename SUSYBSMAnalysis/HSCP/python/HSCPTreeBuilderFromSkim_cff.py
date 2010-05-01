import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff import *

TrackRefitter.src          = "generalTracksSkim"
muontiming.MuonCollection  = cms.InputTag("muonsSkim")
HSCPTreeBuilder.muons      = cms.InputTag("muonsSkim")
