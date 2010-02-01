import FWCore.ParameterSet.Config as cms

process = cms.Process("d0phi")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.load("INPUT_FILE")

process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('LUMIRANGE')
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) #1500
)
process.p = cms.Path(process.d0_phi_analyzer)
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#--------------------------
# Filters
#--------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

# L1 Trigger Bit Selection (bit 40 and 41 for BSC trigger)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41)')
#######################

# run over STA muons
#process.d0_phi_analyzer.BeamFitter.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias') #,'UpdatedAtVtx')
#process.d0_phi_analyzer.BeamFitter.IsMuonCollection = True

process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 6
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 10
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 2
process.d0_phi_analyzer.BeamFitter.MinimumPt = 1.0
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter = 1.0
process.d0_phi_analyzer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
process.d0_phi_analyzer.BeamFitter.InputBeamWidth = 0.0400
process.d0_phi_analyzer.BeamFitter.Debug = True
#########################
process.d0_phi_analyzer.BeamFitter.AsciiFileName = 'ASCIIFILE'
process.d0_phi_analyzer.BeamFitter.OutputFileName = 'OUTPUTFILE'
process.d0_phi_analyzer.BeamFitter.SaveNtuple = True
process.d0_phi_analyzer.BeamFitter.SaveFitResults = True

process.p = cms.Path(process.hltLevel1GTSeed*process.d0_phi_analyzer)
#process.p = cms.Path(process.d0_phi_analyzer)

# fit as function of lumi sections
#process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 2
#process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 10
