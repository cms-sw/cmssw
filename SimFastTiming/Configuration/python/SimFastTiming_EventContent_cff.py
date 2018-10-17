import FWCore.ParameterSet.Config as cms

SimFastTimingFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

SimFastTimingRAW = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

SimFastTimingRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

SimFastTimingAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

SimFastTimingPREMIX = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

_phase2_timing_extraCommands = [ 'keep *_mix_FTLBarrel_*','keep *_mix_FTLEndcap_*','keep *_mix_InitialVertices_*' ]
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( SimFastTimingRAW, outputCommands = SimFastTimingRAW.outputCommands + _phase2_timing_extraCommands )
phase2_timing.toModify( SimFastTimingFEVTDEBUG, outputCommands = SimFastTimingFEVTDEBUG.outputCommands + _phase2_timing_extraCommands )
phase2_timing.toModify( SimFastTimingRECO, outputCommands = SimFastTimingRECO.outputCommands + _phase2_timing_extraCommands )
