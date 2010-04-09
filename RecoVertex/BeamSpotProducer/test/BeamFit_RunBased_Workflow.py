import FWCore.ParameterSet.Config as cms

process = cms.Process("BSworkflow")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/FE0A36FF-11ED-DE11-A8EE-002618943849.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/F89D50B7-07ED-DE11-B079-0026189437ED.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/F0E44CBD-07ED-DE11-A0F7-002618943914.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/DAB587B8-07ED-DE11-8369-00304867C1BC.root',
#    '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/C8E03DBC-07ED-DE11-88ED-00248C0BE005.root'
    '/store/data/BeamCommissioning09/MinimumBias/ALCARECO/StreamTkAlMinBias-Dec19thReReco_341_v1/0006/DEC0A7AE-2CEE-DE11-A801-002618943945.root',
    '/store/data/BeamCommissioning09/MinimumBias/ALCARECO/StreamTkAlMinBias-Dec19thReReco_341_v1/0006/D49332EC-F5ED-DE11-8F55-002618943950.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1) #1500
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

# this is for filtering on L1 technical trigger bit
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 )')
##

## reco PV
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V2::All'

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("ALCARECOTkAlMinBias") 

#### remove beam scraping events
process.noScraping= cms.EDFilter("FilterOutScraping",
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.20)
)

process.p = cms.Path(process.hltLevel1GTSeed +
                     process.offlineBeamSpot +
                     process.offlinePrimaryVertices+
#                     process.noScraping +
                     process.d0_phi_analyzer)

process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

#######################
process.d0_phi_analyzer.BeamFitter.TrackCollection = 'ALCARECOTkAlMinBias'
process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 6
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 10
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 2
process.d0_phi_analyzer.BeamFitter.MinimumPt = 1.0
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter = 1.0
process.d0_phi_analyzer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
process.d0_phi_analyzer.BeamFitter.InputBeamWidth = -1 # 0.0400
process.d0_phi_analyzer.BeamFitter.Debug = True

process.d0_phi_analyzer.PVFitter.Apply3DFit = True
#########################

process.d0_phi_analyzer.BeamFitter.AsciiFileName = 'BeamFit_RunBased_Workflow.txt'
process.d0_phi_analyzer.BeamFitter.AppendRunToFileName = False
process.d0_phi_analyzer.BeamFitter.OutputFileName = 'BeamFit_RunBased_Workflow.root' 
#process.d0_phi_analyzer.BeamFitter.SaveNtuple = True

# fit as function of lumi sections
process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = -1
process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = -1
