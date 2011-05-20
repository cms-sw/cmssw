import FWCore.ParameterSet.Config as cms

from DPGAnalysis.Skims.Skims_DPG_cff import skimContent

from Configuration.EventContent.EventContent_cff import RECOEventContent
skimRecoContent = RECOEventContent.clone()
skimRecoContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimRecoContent.outputCommands.append("drop *_*_*_SKIM")

from Configuration.EventContent.EventContent_cff import AODEventContent
skimAodContent = AODEventContent.clone()
skimAodContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimAodContent.outputCommands.append("drop *_*_*_SKIM")

UnitTestEventContent = cms.PSet(
outputCommands = cms.untracked.vstring('drop *','keep *_TriggerResults_*_*')
    )

from SUSYBSMAnalysis.Skimming.EXOLLResSkim_cff import *
exoLLResmmPath = cms.Path(exoLLResdiMuonSequence)
exoLLReseePath = cms.Path(exoLLResdiElectronSequence)
exoLLResemPath = cms.Path(exoLLResEleMuSequence)
SKIMStreamEXOLLRes = cms.FilteredStream(
        responsible = 'EXO',
        name = 'EXOLLRes',
        paths = (exoLLResmmPath,exoLLReseePath,exoLLResemPath),
        content = skimAodContent.outputCommands,
        selectEvents = cms.untracked.PSet(),
        dataTier = cms.untracked.string('AOD')
        )

from SUSYBSMAnalysis.Skimming.EXOEle_cff import *
exoElePath = cms.Path(exoEleLowetSeqReco)
SKIMStreamEXOEle = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXOEle',
    paths = (exoElePath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from SUSYBSMAnalysis.Skimming.EXOMu_cff import *
exoMuPath = cms.Path(exoMuSequence)
SKIMStreamEXOMu = cms.FilteredStream(
    responsible = 'EXO',
    name = "EXOMu",
    paths = (exoMuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from SUSYBSMAnalysis.Skimming.EXOTriLepton_cff import *
exoTriMuPath = cms.Path(exoTriMuonSequence)
SKIMStreamEXOTriMu = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXOTriMu',
    paths = (exoTriMuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
exoTriElePath = cms.Path(exoTriElectronSequence)
SKIMStreamEXOTriEle = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXOTriEle',
    paths = (exoTriElePath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
exo1E2MuPath = cms.Path(exo1E2MuSequence)
SKIMStreamEXO1E2Mu = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXO1E2Mu',
    paths = (exo1E2MuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )

from SUSYBSMAnalysis.Skimming.EXODiLepton_cff import *
exoDiMuPath = cms.Path(exoDiMuSequence)
exoDiElePath = cms.Path(exoDiMuSequence)
exoEMuPath = cms.Path(exoEMuSequence)
SKIMStreamEXODiMu = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXODiMu',
    paths = (exoDiMuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
SKIMStreamEXODiEle = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXODiEle',
    paths = (exoDiElePath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
SKIMStreamEXOEMu = cms.FilteredStream(
    responsible = 'EXO',
    name = 'EXOEMu',
    paths = (exoEMuPath),
    content = skimAodContent.outputCommands,
    selectEvents = cms.untracked.PSet(),
    dataTier = cms.untracked.string('AOD')
    )
