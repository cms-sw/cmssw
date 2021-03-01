import FWCore.ParameterSet.Config as cms

# The line below always has to be included to make VarParsing work
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing ('analysis')
options.parseArguments()

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


input_filename = 'default.root' if len(options.inputFiles) == 0 else options.inputFiles[0]
#input_filename='step2SingleElectronPt15Eta1p7_2p7_SimTracksters.root'
#input_filename='step2FineCaloSingleElectronPt15Eta1p7_2p7_SimTracksters.root'
#input_filename='step2SingleElectronPt15Eta1p7_2p7_CBWEAndSimTracksters.root'
#input_filename='step2FineCaloSingleElectronPt15Eta1p7_2p7_CBWEAndSimTracksters.root'

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
#        'file:/data/rovere/HGCAL/study/CMSSW_9_4_0/src/SimGeneral/Debugging/test/20088.0_SinglePiPt25Eta1p7_2p7+SinglePiPt25Eta1p7_2p7_2023D17_GenSimHLBeamSpotFull+DigiFullTrigger_2023D17+RecoFullGlobal_2023D17+HARVESTFullGlobal_2023D17/step2.root'
         'file:%s'%input_filename

    )
)

process.load("SimGeneral.Debugging.caloParticleDebugger_cfi")

# MessageLogger customizations
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout.enable = False
labels = ['SimTracks', 'SimVertices', 'GenParticles', 'TrackingParticles', 'CaloParticles', 'SimClusters']
messageLogger = dict()
for category in labels:
    main_key = '%sMessageLogger'%(category)
    category_key = 'CaloParticleDebugger%s'%(category)
    messageLogger[main_key] = dict(
            filename = '%s_%s.log' % (input_filename.replace('.root',''), category),
            threshold = 'INFO',
            default = dict(limit=0)
            )
    messageLogger[main_key][category_key] = dict(limit=-1)
    # First create defaults
    setattr(process.MessageLogger.files, category, dict())
    # Then modify them
    setattr(process.MessageLogger.files, category, messageLogger[main_key])

process.p = cms.Path(process.caloParticleDebugger)
