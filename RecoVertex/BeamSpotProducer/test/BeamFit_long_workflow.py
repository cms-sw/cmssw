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

# this is for filtering on L1 technical trigger bit
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 )')
##

process.p = cms.Path(process.hltLevel1GTSeed + process.d0_phi_analyzer)
##

process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#######################
process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 6
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 10
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 2
process.d0_phi_analyzer.BeamFitter.MinimumPt = 1.0
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter = 1.0 #5.0
process.d0_phi_analyzer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
#process.d0_phi_analyzer.BeamFitter.TrackQuality = cms.untracked.vstring("highPurity")
process.d0_phi_analyzer.BeamFitter.InputBeamWidth = 0.0400
process.d0_phi_analyzer.BeamFitter.Debug = True
#########################

process.d0_phi_analyzer.BeamFitter.AsciiFileName = 'BeamFit_LongWorkflow.txt'
process.d0_phi_analyzer.BeamFitter.OutputFileName = 'BeamSpot_LongWorkflow.root' 
process.d0_phi_analyzer.BeamFitter.SaveNtuple = True

# fit as function of lumi sections
process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 2
process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = -1
