import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.particleFlowDQM_cff import pfJetAnalyzerDQM
from Validation.RecoParticleFlow.particleFlowDQM_cff import pfPuppiJetAnalyzerDQM
from Validation.RecoParticleFlow.particleFlowDQM_cff import pfJetDQMPostProcessor
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetAnalyzerDQM
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetDQMPostProcessor
from Validation.RecoParticleFlow.particleFlowDQM_cff import pfMetAnalyzerDQM
from Validation.RecoParticleFlow.particleFlowDQM_cff import pfPuppiMetAnalyzerDQM
from Validation.RecoTau.RecoTauValidation_cff import *


DQMOfflinePF = cms.Sequence(
  pfJetAnalyzerDQM +
  pfPuppiJetAnalyzerDQM +
  offsetAnalyzerDQM 
)

# MET, Tau, PFCand sequence
DQMOfflinePFExtended = cms.Sequence( 
    pfMetAnalyzerDQM +
    pfPuppiMetAnalyzerDQM +
    tauValidationMiniAODZTT
#    pfCandAnalyzerDQM
)

DQMHarvestPF = cms.Sequence(
  pfJetDQMPostProcessor +
  offsetDQMPostProcessor
)
