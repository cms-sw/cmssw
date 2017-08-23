import FWCore.ParameterSet.Config as cms

from SimMuon.MCTruth.SimMuFilter_cfi import SimMuFilter

SimMuFilter.nMuSel = 1

SimMuFiltSeq = cms.Sequence(SimMuFilter)
