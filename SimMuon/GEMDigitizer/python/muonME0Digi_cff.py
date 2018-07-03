import FWCore.ParameterSet.Config as cms

from SimMuon.GEMDigitizer.muonME0Digis_cfi import *
from SimMuon.GEMDigitizer.muonME0PadDigis_cfi import *
from SimMuon.GEMDigitizer.muonME0PadDigiClusters_cfi import *
from SimMuon.GEMDigitizer.muonME0PseudoDigis_cfi import *
from SimMuon.GEMDigitizer.muonME0PseudoReDigis_cfi import *

muonME0RealDigi = cms.Sequence(simMuonME0Digis * simMuonME0PadDigis + simMuonME0PadDigiClusters)

muonME0PseudoDigi = cms.Sequence(simMuonME0PseudoDigis * simMuonME0PseudoReDigis)

muonME0Digi = cms.Sequence(muonME0RealDigi * muonME0PseudoDigi)

# In premixing we use simMuonME0Digis for the overlay, and the rest are run after that
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toReplaceWith(muonME0Digi, cms.Sequence(simMuonME0Digis))

muonME0DigiDM = cms.Sequence(simMuonME0PadDigis + simMuonME0PadDigiClusters + muonME0PseudoDigi)
