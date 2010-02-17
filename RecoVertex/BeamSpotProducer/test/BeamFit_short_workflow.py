import FWCore.ParameterSet.Config as cms

process = cms.Process("BSworkflow")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) #1500
)

process.p = cms.Path(process.d0_phi_analyzer)
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#######################
#process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 15
#process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
#process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 20
#process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 1
#########################

process.d0_phi_analyzer.BeamFitter.AsciiFileName = 'BeamFit_ShortWorkflow.txt'
process.d0_phi_analyzer.BeamFitter.OutputFileName = 'BeamSpot_ShortWorkflow.root' 
process.d0_phi_analyzer.BeamFitter.SaveNtuple = True

# fit as function of lumi sections
process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 2
process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 10
