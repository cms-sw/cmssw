import FWCore.ParameterSet.Config as cms

from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_Client_cfi import *

postProcessorHLTvertexing = postProcessorVertex.clone()
postProcessorHLTvertexing.subDirs = cms.untracked.vstring("HLT/Vertexing/ValidationWRTsim/*")
