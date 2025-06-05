import FWCore.ParameterSet.Config as cms


def embeddingHLTCustomiser(process):
    """
    Function to modify the HLT configuration for embedding.
    """
    # Replace the online beam spot producer with the embedding version
    process.hltOnlineBeamSpot = cms.EDProducer('EmbeddingBeamSpotOnlineProducer', src = cms.InputTag('offlineBeamSpot'))
    process.hltPixelVertices  = cms.EDProducer('EmbeddingHltPixelVerticesProducer')
    # Replace HLT vertexing with vertex taken from LHE step
    process.load("TauAnalysis.MCEmbeddingTools.EmbeddingHltPixelVerticesProducer_cfi")
    process.hltPixelVertices = cms.EDProducer('EmbeddingHltPixelVerticesProducer')
    process.offlinePrimaryVertices = cms.EDProducer('EmbeddingHltPixelVerticesProducer')
    process.firstStepPrimaryVerticesUnsorted = cms.EDProducer('EmbeddingHltPixelVerticesProducer')
    process.firstStepPrimaryVerticesPreSplitting = cms.EDProducer('EmbeddingHltPixelVerticesProducer')
    
    # Disable the original detector state filters in the HLT step.
    # This is done by replacing them with one that always passes (100% efficiency).
    # Those original filters have a efficiency of 0% for embedding samples, due to the fact 
    # that the simulation of the tau decay happens in an empty detector.
    # For more info see https://github.com/cms-sw/cmssw/pull/47299#discussion_r1949023230
    process.hltPixelTrackerHVOn = cms.EDFilter("HLTBool", result = cms.bool(True))

    process.hltStripTrackerHVOn = cms.EDFilter("HLTBool", result = cms.bool(True))

    return process