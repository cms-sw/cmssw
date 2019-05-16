import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.particleFlowDQM_cff import pfDQMAnalyzer
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetAnalyzerDQM

DQMOfflinePF = cms.Sequence(
  pfDQMAnalyzer +
  offsetAnalyzerDQM
)

