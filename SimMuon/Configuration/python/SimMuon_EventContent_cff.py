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

# The merged SimDigi collections have mixData label, drop the signal-only ones
# Eventually (if/when we switch muon premixing to use SimHits) this whole customization
#   can be removed as the collection names become the standard ones
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(SimMuonFEVTDEBUG, outputCommands = [
        'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_mixData_*_*',
        'keep CSCDetIdCSCStripDigiMuonDigiCollection_mixData_*_*',
        'keep CSCDetIdCSCWireDigiMuonDigiCollection_mixData_*_*',
        'keep DTLayerIdDTDigiMuonDigiCollection_mixData_*_*',
        'keep RPCDetIdRPCDigiMuonDigiCollection_mixData_*_*',
])
# Almost same (without DT) for SimMuonRAW following the run2_common customization
premix_stage2.toModify(SimMuonRAW, outputCommands = [
        'keep *_mixData_MuonCSCStripDigiSimLinks_*',
        'keep DTLayerIdDTDigiSimLinkMuonDigiCollection_mixData_*_*',
        'keep *_mixData_RPCDigiSimLink_*',
])
(run2_common & premix_stage2).toModify(SimMuonRAW, outputCommands = SimMuonRAW.outputCommands + [
        'drop *_simMuonCSCDigis_*_*',
        'drop *_simMuonRPCDigis_*_*',
        'keep *_mixData_MuonCSCWireDigiSimLinks_*',
        'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_mixData_*_*',
        'keep CSCDetIdCSCStripDigiMuonDigiCollection_mixData_*_*',
        'keep CSCDetIdCSCWireDigiMuonDigiCollection_mixData_*_*',
        'keep RPCDetIdRPCDigiMuonDigiCollection_mixData_*_*',
])

#RECO content
SimMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep StripDigiSimLinkedmDetSetVector_simMuonCSCDigis_*_*', 
        'keep DTLayerIdDTDigiSimLinkMuonDigiCollection_simMuonDTDigis_*_*', 
        'keep RPCDigiSimLinkedmDetSetVector_simMuonRPCDigis_*_*')
)
# The merged DigiSimLink collections have mixData label, drop the signal-only ones
premix_stage2.toModify(SimMuonRECO, outputCommands = [
        'keep *_mixData_MuonCSCStripDigiSimLinks_*',
        'keep *_mixData_MuonCSCWireDigiSimLinks_*',
        'keep *_mixData_RPCDigiSimLink_*',
        'keep DTLayerIdDTDigiSimLinkMuonDigiCollection_mixData_*_*',
])

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
        'keep *_simMuonDTDigis_*_*',
        'keep *_simMuonCSCDigis_*_*',
        'keep *_simMuonCscTriggerPrimitiveDigis_*_*',
        'keep *_simMuonRPCDigis_*_*',
    )
)

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toModify( SimMuonFEVTDEBUG, outputCommands = SimMuonFEVTDEBUG.outputCommands + ['keep *_simMuonGEMDigis_*_*',
                                                                                              'keep *_simMuonGEMPadDigis_*_*',
                                                                                              'keep *_simMuonGEMPadDigiClusters_*_*'] )
run2_GEM_2017.toModify( SimMuonRAW, outputCommands = SimMuonRAW.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*'] )
run2_GEM_2017.toModify( SimMuonRECO, outputCommands = SimMuonRECO.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*'] )
run2_GEM_2017.toModify( SimMuonPREMIX, outputCommands = SimMuonPREMIX.outputCommands + ['keep *_simMuonGEMDigis_*_*',
                                                                                        'keep *_*_GEMDigiSimLink_*',
                                                                                        'keep *_*_GEMStripDigiSimLink_*'] )


from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( SimMuonFEVTDEBUG, outputCommands = SimMuonFEVTDEBUG.outputCommands + ['keep *_simMuonGEMDigis_*_*',
                                                                                         'keep *_simMuonGEMPadDigis_*_*',
                                                                                         'keep *_simMuonGEMPadDigiClusters_*_*'] )
run3_GEM.toModify( SimMuonRAW, outputCommands = SimMuonRAW.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*'] )
run3_GEM.toModify( SimMuonRECO, outputCommands = SimMuonRECO.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonGEMDigis_*_*'] )
run3_GEM.toModify( SimMuonPREMIX, outputCommands = SimMuonPREMIX.outputCommands + ['keep *_simMuonGEMDigis_*_*',
                                                                                   'keep *_*_GEMDigiSimLink_*',
                                                                                   'keep *_*_GEMStripDigiSimLink_*'] )

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify( SimMuonFEVTDEBUG, outputCommands = SimMuonFEVTDEBUG.outputCommands + ['keep *_simMuonME0PseudoDigis_*_*',
                                                                                            'keep *_simMuonME0PseudoReDigis_*_*',
                                                                                            'keep *_simMuonME0Digis_*_*',
                                                                                            'keep *_simMuonME0PadDigis_*_*',
                                                                                            'keep *_simMuonME0PadDigiClusters_*_*'] )
phase2_muon.toModify( SimMuonRAW, outputCommands = SimMuonRAW.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonME0Digis_*_*'] )
phase2_muon.toModify( SimMuonRECO, outputCommands = SimMuonRECO.outputCommands + ['keep StripDigiSimLinkedmDetSetVector_simMuonME0Digis_*_*'] )
phase2_muon.toModify( SimMuonPREMIX, outputCommands = SimMuonPREMIX.outputCommands + ['keep *_simMuonME0Digis_*_*',
                                                                                      'keep *_mix_g4SimHitsMuonME0Hits_*',
                                                                                      'keep *_*_ME0DigiSimLink_*',
                                                                                      'keep *_*_ME0StripDigiSimLink_*'] )


# For phase2 premixing switch the sim digi collections to the ones including pileup
(premix_stage2 & phase2_muon).toModify(SimMuonFEVTDEBUG, outputCommands = SimMuonFEVTDEBUG.outputCommands + [
    'drop *_simMuonGEMDigis_*_*',
    'keep GEMDetIdGEMDigiMuonDigiCollection_mixData_*_*',
    'drop *_simMuonME0Digis_*_*',
    'keep ME0DetIdME0DigiMuonDigiCollection_mixData_*_*',
])
