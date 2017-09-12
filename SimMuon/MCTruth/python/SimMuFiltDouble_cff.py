import FWCore.ParameterSet.Config as cms

from SimMuon.MCTruth.SimMuFilter_cfi import SimMuFilter

SimDoubleMuFilter = SimMuFilter.clone()
SimDoubleMuFilter.nMuSel = 2

SimMuFiltSeq = cms.Sequence(SimDoubleMuFilter)
