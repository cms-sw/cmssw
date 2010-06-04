import FWCore.ParameterSet.Config as cms

process = cms.Process("BSworkflow")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/D89CFC67-3D67-DF11-BB3C-001D09F25393.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/D6578636-3B67-DF11-866E-001D09F23D1D.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/B8399269-3D67-DF11-AA32-001D09F2AD84.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/B81E3C6E-3B67-DF11-9903-001D09F27067.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/8A3A09B7-3C67-DF11-936C-001D09F2438A.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/82850560-3B67-DF11-AE51-001D09F253FC.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/44CB1DDD-3E67-DF11-B462-0019B9F70607.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/369DDB31-3B67-DF11-8681-001D09F23944.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/0405EBB8-3C67-DF11-A97E-001D09F2A690.root',
'/store/express/Run2010A/StreamExpress/ALCARECO/v2/000/136/087/0067FCDD-3E67-DF11-A1F4-001D09F2AF1E.root'

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
    input = cms.untracked.int32(5) #1500
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

# fit as function of lumi sections
process.d0_phi_analyzer.BSAnalyzerParameters.fitEveryNLumi = 1
process.d0_phi_analyzer.BSAnalyzerParameters.resetEveryNLumi = 1
