import FWCore.ParameterSet.Config as cms

process = cms.Process("BSworkflow")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0171/3A22D08D-456A-DF11-961F-001A92811734.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0171/083B6802-236C-DF11-8AC6-0026189437FE.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0170/E6D0589B-136A-DF11-9E90-002618943982.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0170/D098488B-276A-DF11-8069-003048678AF4.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/F466DD94-BF69-DF11-B9B8-00261894390A.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/F2A9245B-BC69-DF11-9C3F-0018F3D096E4.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/DED5E502-B969-DF11-AE8F-002618943964.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/563257AF-C169-DF11-865F-002618943907.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/3056B847-C069-DF11-B0B0-003048679010.root'

    )
)

process.MessageLogger.cerr.FwkReport  = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1000000),
)

#process.source = cms.Source('PoolSource',
#                            debugVerbosity = cms.untracked.uint32(0),
#                            debugFlag = cms.untracked.bool(False)
#                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100) #1500
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
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_36X_V11::All' #'GR_R_35X_V8::All'
process.load("Configuration.StandardSequences.Geometry_cff")


########## RE-FIT TRACKS
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = 'ALCARECOTkAlMinBias'

## reco PV
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("TrackRefitter") 

#### remove beam scraping events
process.noScraping= cms.EDFilter("FilterOutScraping",
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.20)
)

process.p = cms.Path(process.hltLevel1GTSeed +
                     process.offlineBeamSpot +
                     process.TrackRefitter +
                     process.offlinePrimaryVertices+
#                     process.noScraping +
                     process.d0_phi_analyzer)

process.MessageLogger.debugModules = ['BeamSpotAnalyzer']

################### Primary Vertex
process.offlinePrimaryVertices.PVSelParameters.maxDistanceToBeam = 2
process.offlinePrimaryVertices.TkFilterParameters.maxNormalizedChi2 = 20
process.offlinePrimaryVertices.TkFilterParameters.minSiliconLayersWithHits = 5
process.offlinePrimaryVertices.TkFilterParameters.maxD0Significance = 100
process.offlinePrimaryVertices.TkFilterParameters.minPixelLayersWithHits = 1
process.offlinePrimaryVertices.TkClusParameters.TkGapClusParameters.zSeparation = 1


#######################
process.d0_phi_analyzer.BeamFitter.TrackCollection = 'TrackRefitter'
process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 6
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 10
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 50
process.d0_phi_analyzer.BeamFitter.MinimumPt = 1.0
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter = 1.0
process.d0_phi_analyzer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
#process.d0_phi_analyzer.BeamFitter.Debug = True

process.d0_phi_analyzer.PVFitter.Apply3DFit = True
process.d0_phi_analyzer.PVFitter.minNrVerticesForFit = 10 
#########################

process.d0_phi_analyzer.BeamFitter.AsciiFileName = 'BeamFit_LumiBased_NewAlignWorkflow.txt'
process.d0_phi_analyzer.BeamFitter.AppendRunToFileName = False
process.d0_phi_analyzer.BeamFitter.OutputFileName = 'BeamFit_LumiBased_Workflow.root' 
#process.d0_phi_analyzer.BeamFitter.SaveNtuple = True
#process.d0_phi_analyzer.BeamFitter.SavePVVertices = True

# fit as function of lumi sections
process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 1
process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 1
