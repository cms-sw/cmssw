import FWCore.ParameterSet.Config as cms

from SimMuon.MCTruth.SimMuFilter_cfi import SimMuFilter

SimSingleMuFilter = SimMuFilter.clone()
SimSingleMuFilter.nMuSel = 1

SimMuFiltSeq = cms.Sequence(SimSingleMuFilter)
