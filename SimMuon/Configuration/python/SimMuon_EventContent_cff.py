# The following comments couldn't be translated into the new config version:

#save digis

#save digis sim link and trigger infos

import FWCore.ParameterSet.Config as cms

# Used to make conditional changes for different running scenarios

#Full Event content with DIGI
SimMuonFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_simMuonCSCDigis_*_*', 
        'keep *_simMuonDTDigis_*_*', 
        'keep *_simMuonRPCDigis_*_*')
)
#RAW content
SimMuonRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep StripDigiSimLinkedmDetSetVector_simMuonCSCDigis_*_*', 
        'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_simMuonCSCDigis_*_*', 
        'keep DTLayerIdDTDigiSimLinkMuonDigiCollection_simMuonDTDigis_*_*', 
        'keep RPCDigiSimLinkedmDetSetVector_simMuonRPCDigis_*_*')
)
# Add extra collections if running in Run 2. Not sure why but these
# collections were added to pretty much all event content in the old
# customisation function.
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( SimMuonRAW.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_simMuonCSCDigis_*_*') )
run2_common.toModify( SimMuonRAW.outputCommands, func=lambda outputCommands: outputCommands.append('keep *_simMuonRPCDigis_*_*') )

#RECO content
SimMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep StripDigiSimLinkedmDetSetVector_simMuonCSCDigis_*_*', 
        'keep DTLayerIdDTDigiSimLinkMuonDigiCollection_simMuonDTDigis_*_*', 
        'keep RPCDigiSimLinkedmDetSetVector_simMuonRPCDigis_*_*')
)

#AOD content
SimMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        # Sim matching information
        'keep *_muonSimClassifier_*_*'
        )
)
SimMuonRECO.outputCommands.extend(SimMuonAOD.outputCommands)

# Event content for premixing library
SimMuonPREMIX = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_mix_g4SimHitsMuonDTHits_*',
        'keep *_mix_g4SimHitsMuonCSCHits_*',
        'keep *_mix_g4SimHitsMuonRPCHits_*',
    )
)

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toModify( SimMuonFEVTDEBUG, outputCommands = SimMuonFEVTDEBUG.outputCommands + ['keep *_simMuonGEMDigis_*_*',
                                                                                              'keep *_simMuonGEMPadDigis_*_*',
                                                                                              'keep *_simMuonGEMPadDigiClusters_*_*'] )
run2_GEM_2017.toModify( SimMuonRAW, outputCommands = SimMuonRAW.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*'] )
run2_GEM_2017.toModify( SimMuonRECO, outputCommands = SimMuonRECO.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*'] )
run2_GEM_2017.toModify( SimMuonPREMIX, outputCommands = SimMuonPREMIX.outputCommands + ['keep *_mix_g4SimHitsMuonGEMHits_*'] )


from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( SimMuonFEVTDEBUG, outputCommands = SimMuonFEVTDEBUG.outputCommands + ['keep *_simMuonGEMDigis_*_*',
                                                                                         'keep *_simMuonGEMPadDigis_*_*',
                                                                                         'keep *_simMuonGEMPadDigiClusters_*_*'] )
run3_GEM.toModify( SimMuonRAW, outputCommands = SimMuonRAW.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*'] )
run3_GEM.toModify( SimMuonRECO, outputCommands = SimMuonRECO.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*'] )
run3_GEM.toModify( SimMuonPREMIX, outputCommands = SimMuonPREMIX.outputCommands + ['keep *_mix_g4SimHitsMuonGEMHits_*'] )

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify( SimMuonFEVTDEBUG, outputCommands = SimMuonFEVTDEBUG.outputCommands + ['keep *_simMuonME0PseudoDigis_*_*',
                                                                                            'keep *_simMuonME0PseudoReDigis_*_*',
                                                                                            'keep *_simMuonME0Digis_*_*',
                                                                                            'keep *_simMuonME0PadDigis_*_*',
                                                                                            'keep *_simMuonME0PadDigiClusters_*_*'] )
phase2_muon.toModify( SimMuonRAW, outputCommands = SimMuonRAW.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonME0Digis_*_*'] )
phase2_muon.toModify( SimMuonRECO, outputCommands = SimMuonRECO.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonME0Digis_*_*'] )
phase2_muon.toModify( SimMuonPREMIX, outputCommands = SimMuonPREMIX.outputCommands + ['keep *_mix_g4SimHitsMuonME0Hits_*'] )


# FastSim uses different naming convention
from Configuration.Eras.Modifier_fastSim_cff import fastSim
def _renameForFastsim(s):
    return s.replace("_g4Sim", "_MuonSim")
fastSim.toModify(SimMuonPREMIX, outputCommands = map(_renameForFastsim, SimMuonPREMIX.outputCommands.value()))
