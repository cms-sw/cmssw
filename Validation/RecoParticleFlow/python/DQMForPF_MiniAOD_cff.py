import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.particleFlowDQM_cff import pfJetAnalyzerDQM
from Validation.RecoParticleFlow.particleFlowDQM_cff import pfPuppiJetAnalyzerDQM
from Validation.RecoParticleFlow.particleFlowDQM_cff import pfJetDQMPostProcessor
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetAnalyzerDQM
from Validation.RecoParticleFlow.offsetAnalyzerDQM_cff import offsetDQMPostProcessor
# Use also other POGs' analyzers for extended checks
from Validation.RecoMET.METRelValForDQM_cff import *
from Validation.RecoJets.JetValidation_cff import *

DQMOfflinePF = cms.Sequence(
  pfJetAnalyzerDQM +
  pfPuppiJetAnalyzerDQM +
  offsetAnalyzerDQM +
#  pfCandAnalyzerDQM +
# Extended ones from other POGs
  METValidationMiniAOD +
  JetValidationMiniAOD
)

DQMHarvestPF = cms.Sequence(
  pfJetDQMPostProcessor +
  offsetDQMPostProcessor
)
