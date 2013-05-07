import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration
from DQMOffline.RecoB.bTagCombinedSVVariables_cff import *
#includes added because of block refactoring replacing simple includes by using statements
from DQMOffline.RecoB.bTagTrackIPAnalysis_cff import *
from DQMOffline.RecoB.bTagCombinedSVAnalysis_cff import *
from DQMOffline.RecoB.bTagTrackCountingAnalysis_cff import *
from DQMOffline.RecoB.bTagTrackProbabilityAnalysis_cff import *
from DQMOffline.RecoB.bTagTrackBProbabilityAnalysis_cff import *
from DQMOffline.RecoB.bTagGenericAnalysis_cff import *
from DQMOffline.RecoB.bTagSimpleSVAnalysis_cff import *
from DQMOffline.RecoB.bTagSoftLeptonAnalysis_cff import *
from DQMOffline.RecoB.bTagSoftLeptonByPtAnalysis_cff import *
from DQMOffline.RecoB.bTagSoftLeptonByIPAnalysis_cff import *
from DQMOffline.RecoB.bTagCommon_cff import *
bTagValidation = cms.EDAnalyzer("BTagPerformanceAnalyzerMC",
    bTagCommonBlock,
    finalizeOnly = cms.bool(False),
    applyPtHatWeight = cms.bool(False),
    jetCorrection = cms.string(''),
    recJetMatching = cms.PSet(
        refJetCorrection = cms.string(''),
        recJetCorrection = cms.string(''),
        maxChi2 = cms.double(50),
        # Corrected calo jets
        sigmaDeltaR = cms.double(0.1),
        sigmaDeltaE = cms.double(0.15)
    ),
    finalizePlots = cms.bool(True),
    tagConfig = cms.VPSet(
        cms.PSet(
            bTagTrackIPAnalysisBlock,
            type = cms.string('TrackIP'),
            label = cms.InputTag("impactParameterTagInfos"),
            folder = cms.string("IPTag")
        ), 
        cms.PSet(
            bTagCombinedSVAnalysisBlock,
            ipTagInfos = cms.InputTag("impactParameterTagInfos"),
            type = cms.string('GenericMVA'),
            svTagInfos = cms.InputTag("secondaryVertexTagInfos"),
            label = cms.InputTag("combinedSecondaryVertex"),
            folder = cms.string("CSVTag")
        ), 
        cms.PSet(
            bTagTrackCountingAnalysisBlock,
            label = cms.InputTag("trackCountingHighEffBJetTags"),
            folder = cms.string("TCHE")
        ), 
        cms.PSet(
            bTagTrackCountingAnalysisBlock,
            label = cms.InputTag("trackCountingHighPurBJetTags"),
            folder = cms.string("TCHP")
        ), 
        cms.PSet(
            bTagProbabilityAnalysisBlock,
            label = cms.InputTag("jetProbabilityBJetTags"),
            folder = cms.string("JP")
        ), 
        cms.PSet(
            bTagBProbabilityAnalysisBlock,
            label = cms.InputTag("jetBProbabilityBJetTags"),
            folder = cms.string("JBP")
        ), 
        cms.PSet(
            bTagSimpleSVAnalysisBlock,
            label = cms.InputTag("simpleSecondaryVertexHighEffBJetTags"),
            folder = cms.string("SSVHE")
        ), 
        cms.PSet(
            bTagSimpleSVAnalysisBlock,
            label = cms.InputTag("simpleSecondaryVertexHighPurBJetTags"),
            folder = cms.string("SSVHP")
        ), 
        cms.PSet(
            bTagGenericAnalysisBlock,
            label = cms.InputTag("combinedSecondaryVertexBJetTags"),
            folder = cms.string("CSV")
        ), 
        cms.PSet(
            bTagGenericAnalysisBlock,
            label = cms.InputTag("combinedSecondaryVertexMVABJetTags"),
            folder = cms.string("CSVMVA")
        ), 
        cms.PSet(
            bTagGenericAnalysisBlock,
            label = cms.InputTag("ghostTrackBJetTags"),
            folder = cms.string("GhTrk")
        ), 
        cms.PSet(
            bTagSoftLeptonAnalysisBlock,
            label = cms.InputTag("softPFMuonBJetTags"),
            folder = cms.string("SMT")
        ), 
        cms.PSet(
            bTagSoftLeptonAnalysisBlock,
            label = cms.InputTag("softPFElectronBJetTags"),
            folder = cms.string("SET")
        ), 
    ),

    flavPlots = cms.string("allbcl"),                            
    differentialPlots = cms.bool(False), #not needed in validation procedure, put True to produce them  
    leptonPlots = cms.uint32(0)
)
