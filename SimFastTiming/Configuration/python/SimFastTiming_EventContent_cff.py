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

_phase2_timing_extraCommands = cms.PSet( # using PSet in order to customize with Modifier
    value = cms.vstring( 'keep *_mix_FTLBarrel_*','keep *_mix_FTLEndcap_*','keep *_mix_InitialVertices_*' )
)
# For premixing switch the sim digi collections to the ones including pileup
# Unsure what to do with InitialVertices, they don't seem to be consumed downstream?
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(_phase2_timing_extraCommands,
    value = [ 'keep *_mixData_FTLBarrel_*','keep *_mixData_FTLEndcap_*','keep *_mix_InitialVertices_*' ]
)
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( SimFastTimingRAW, outputCommands = SimFastTimingRAW.outputCommands + _phase2_timing_extraCommands.value )
phase2_timing.toModify( SimFastTimingFEVTDEBUG, outputCommands = SimFastTimingFEVTDEBUG.outputCommands + _phase2_timing_extraCommands.value )
phase2_timing.toModify( SimFastTimingRECO, outputCommands = SimFastTimingRECO.outputCommands + _phase2_timing_extraCommands.value )
phase2_timing.toModify( SimFastTimingPREMIX, outputCommands = SimFastTimingRECO.outputCommands + _phase2_timing_extraCommands.value )
