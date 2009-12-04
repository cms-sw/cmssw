import FWCore.ParameterSet.Config as cms

process = cms.Process("d0phi")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	#'file:FirstEvent.root'
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_DESIGN_3X_V8A_v1/0082/22F2A8A8-8BD8-DE11-A2FE-00248C0BE01E.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_DESIGN_3X_V8A_v1/0082/3CF7DD76-8CD8-DE11-9C9A-0026189438D5.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_DESIGN_3X_V8A_v1/0082/66273AA8-8BD8-DE11-A9FA-0026189438BC.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_DESIGN_3X_V8A_v1/0082/988EDEDB-8DD8-DE11-8E82-00261894386D.root'

#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_STARTUP3X_V8D_v1/0082/16260D10-89D8-DE11-9578-0026189437E8.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_STARTUP3X_V8D_v1/0082/18DD5AEF-89D8-DE11-88AF-002618943956.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_STARTUP3X_V8D_v1/0082/488187EE-89D8-DE11-957D-00248C0BE01E.root',
#'/store/mc/Summer09/MinBias/ALCARECO/StreamTkAlMinBias-334_STARTUP3X_V8D_v1/0082/5E1915A7-8BD8-DE11-8C3D-002618943956.root'
#	'file:BSCskim_123151_Express.root'
    'file:Run122314_BSCSkim_MinBiasPD_ReTracking.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) #1500
)
process.p = cms.Path(process.d0_phi_analyzer)
process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#######################

# run over STA muons
#process.d0_phi_analyzer.BeamFitter.TrackCollection = cms.untracked.InputTag('ALCARECOTkAlMinBias') #,'UpdatedAtVtx')
#process.d0_phi_analyzer.BeamFitter.IsMuonCollection = True

process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 6
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 10
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 2
process.d0_phi_analyzer.BeamFitter.MinimumPt = 1.0
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter = 5.0
process.d0_phi_analyzer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
process.d0_phi_analyzer.BeamFitter.Debug = True
#########################

process.d0_phi_analyzer.BeamFitter.OutputFileName = 'run122314.root' #AtVtx10000.root'
process.d0_phi_analyzer.BeamFitter.SaveNtuple = True

# fit as function of lumi sections
#process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 2
#process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 10
