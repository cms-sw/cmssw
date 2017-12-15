import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    inputCommands = cms.untracked.vstring(['keep *',
                                           'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
                                           'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
                                           'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
                                           'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
                                           'drop l1tEMTFTrack2016s_simEmtfDigis__HLT']),
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
#      'file:/data/rovere/HGCAL/study/CMSSW_9_4_0/src/SimGeneral/Debugging/test/20800.0_FourMuPt1_200+FourMuPt_1_200_pythia8_2023D20_GenSimHLBeamSpotFull+DigiFull_2023D20+RecoFullGlobal_2023D20+HARVESTFullGlobal_2023D20/step2.root'
#        'file:/data/rovere/HGCAL/study/CMSSW_9_4_0/src/SimGeneral/Debugging/test/20824.0_TTbar_13+TTbar_13TeV_TuneCUETP8M1_2023D20_GenSimHLBeamSpotFull+DigiFull_2023D20+RecoFullGlobal_2023D20+HARVESTFullGlobal_2023D20/step2.root'
#        'file:/data/rovere/HGCAL/study/CMSSW_9_4_0/src/SimGeneral/Debugging/test/20002.0_SingleElectronPt35+SingleElectronPt35_pythia8_2023D17_GenSimHLBeamSpotFull+DigiFullTrigger_2023D17+RecoFullGlobal_2023D17+HARVESTFullGlobal_2023D17/step2.root'
#        'file:/data/rovere/HGCAL/study/CMSSW_9_4_0/src/SimGeneral/Debugging/test/20016.0_SingleGammaPt35Extended+DoubleGammaPt35Extended_pythia8_2023D17_GenSimHLBeamSpotFull+DigiFullTrigger_2023D17+RecoFullGlobal_2023D17+HARVESTFullGlobal_2023D17/step2.root'
        'file:/data/rovere/HGCAL/study/CMSSW_9_4_0/src/SimGeneral/Debugging/test/20088.0_SinglePiPt25Eta1p7_2p7+SinglePiPt25Eta1p7_2p7_2023D17_GenSimHLBeamSpotFull+DigiFullTrigger_2023D17+RecoFullGlobal_2023D17+HARVESTFullGlobal_2023D17/step2.root'
    )
)

process.load("SimGeneral.Debugging.caloParticleDebugger_cfi")

process.p = cms.Path(process.caloParticleDebugger)
