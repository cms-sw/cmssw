import FWCore.ParameterSet.Config as cms

# File: CaloMET.cfi
# Author: B. Scurlock & R. Remington
# Date: 03.04.2008
#
# Fill validation histograms for MET
metAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("met")
    )

metHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metHO")
    )

metNoHFAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metNoHF")
    )

metNoHFHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metNoHFHO")
    )

metOptAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metOpt")
    )

metOptHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metOptHO")
    )

metOptNoHFAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metOptNoHF")
    )

metOptNoHFHOAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("metOptNoHFHO")
    )

pfMetAnalyzer = cms.EDAnalyzer(
   "METTester",
   OutputFile = cms.untracked.string('METTester.root'),
   InputMETLabel = cms.InputTag("pfMet")
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
    sample = cms.untracked.string('NULL'),
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
    ) 


genMptTrueAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMptTrue"),
    )

genMetTrueAnalyzer = cms.EDAnalyzer(
    "METTester",
   OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMetTrue")
    )

genMetCaloAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMetCalo")
    )

genMptCaloAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMptCalo")
    )


genMetCaloAndNonPromptAnalyzer = cms.EDAnalyzer(
    "METTester",
    OutputFile = cms.untracked.string('METTester.root'),
    InputMETLabel = cms.InputTag("genMetCaloAndNonPrompt")
    )

