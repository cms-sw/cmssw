import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.particleFlowDQM_cff import pfJetAnalyzerDQM
from Validation.RecoParticleFlow.particleFlowDQM_cff import pfJetDQMPostProcessor
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetAnalyzerDQM
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetDQMPostProcessor

DQMOfflinePF = cms.Sequence(
  pfJetAnalyzerDQM +
  offsetAnalyzerDQM
)

DQMHarvestPF = cms.Sequence(
  pfJetDQMPostProcessor +
  offsetDQMPostProcessor
)
