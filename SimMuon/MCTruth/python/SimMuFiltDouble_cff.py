import FWCore.ParameterSet.Config as cms

from SimMuon.MCTruth.SimMuFilter_cfi import SimMuFilter

SimMuFilter.nMuSel = 2

SimMuFiltSeq = cms.Sequence(SimMuFilter)
