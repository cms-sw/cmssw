import FWCore.ParameterSet.Config as cms

from SimGeneral.MuonAnalysis.SimMuFilter_cfi import *

process = cms.Process("SimFilter")

process.filter = SimMuFilter

process.p = cms.Path(process.filter)
