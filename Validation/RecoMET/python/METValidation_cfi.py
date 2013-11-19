import FWCore.ParameterSet.Config as cms

# File: CaloMET.cfi
# Author: B. Scurlock & R. Remington
# Date: 03.04.2008
#
# Fill validation histograms for MET
metAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("met"),
    METType = cms.untracked.string('CaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metHO"),
    METType = cms.untracked.string('CaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metNoHFAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metNoHF"),
    METType = cms.untracked.string('CaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metNoHFHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metNoHFHO"),
    METType = cms.untracked.string('CaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metOptAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metOpt"),
    METType = cms.untracked.string('CaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metOptHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metOptHO"),
    METType = cms.untracked.string('CaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

metOptNoHFAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metOptNoHF"),
    METType = cms.untracked.string('CaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")  
    )

metOptNoHFHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metOptNoHFHO"),
    METType = cms.untracked.string('CaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

pfMetAnalyzer = cms.EDAnalyzer(
   "METTester",
   OutputFile = cms.untracked.string('METTester.root'),
   InputMETLabel = cms.InputTag("pfMet"),
   METType = cms.untracked.string('PFMET')
   #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
   ) 

tcMetAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("tcMet"),
    InputCaloMETLabel = cms.InputTag("met"),
    InputTrackLabel = cms.InputTag("generalTracks"),
    InputMuonLabel = cms.InputTag("muons"),
    InputElectronLabel = cms.InputTag("gsfElectrons"),
    InputBeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    METType = cms.untracked.string('TCMET'),
    sample = cms.untracked.string('NULL'),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/"),
    minhits = cms.int32(6),
    maxd0 = cms.double(0.1),
    maxchi2 = cms.double(5),
    maxeta = cms.double(2.65),
    maxpt = cms.double(100.),
    maxPtErr = cms.double(0.2),
    trkQuality = cms.vint32(2),
    trkAlgos = cms.vint32()
    ) 

corMetGlobalMuonsAnalyzer = cms.EDAnalyzer(
    "METTester",
   OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("corMetGlobalMuons"),
    METType = cms.untracked.string('MuonCorrectedCaloMET')
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    ) 


genMptTrueAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMptTrue"),
    METType = cms.untracked.string("GenMET")
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

genMetTrueAnalyzer = cms.EDAnalyzer(
    "METTester",
   OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMetTrue"),
    METType = cms.untracked.string("GenMET")
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

genMetCaloAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMetCalo"),
    METType = cms.untracked.string("GenMET")
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

genMptCaloAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMptCalo"),
    METType = cms.untracked.string("GenMET")
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )


genMetCaloAndNonPromptAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMetCaloAndNonPrompt"),
    METType = cms.untracked.string("GenMET")
    #FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    )

