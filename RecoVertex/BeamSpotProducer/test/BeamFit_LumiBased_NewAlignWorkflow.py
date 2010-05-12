import FWCore.ParameterSet.Config as cms

process = cms.Process("BSworkflow")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/express/Commissioning10/StreamExpress/ALCARECO/v9/000/135/149/F053205D-8E5B-DF11-803D-000423D9997E.root',
'/store/express/Commissioning10/StreamExpress/ALCARECO/v9/000/135/149/F038B0BE-3A5B-DF11-B94E-000423D98750.root',
'/store/express/Commissioning10/StreamExpress/ALCARECO/v9/000/135/149/F03560AE-165B-DF11-AE55-0030486780EC.root',
'/store/express/Commissioning10/StreamExpress/ALCARECO/v9/000/135/149/F03169C7-935B-DF11-A59F-001D09F28D54.root',
'/store/express/Commissioning10/StreamExpress/ALCARECO/v9/000/135/149/EE15D8B9-1E5B-DF11-8205-003048D2BC30.root'

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
process.GlobalTag.globaltag = 'GR10_P_V5::All' #'GR_R_35X_V8::All'
process.load("Configuration.StandardSequences.Geometry_cff")


#############  NEW ALIGNMENT ###########################################################
### begin of corrections to strip local reco - needed only if you run on TK-DECO data (i.e., MinBias) !!!! For cosmics, comment them out
# set of corrections (backplane and Lorentz angle) from the Strip Local Calibration group by G.Kaussen
# http://indico.cern.ch/getFile.py/access?subContId=0&contribId=1&resId=0&materialId=slides&confId=93637
#############################################################################################################

## backplane corrections
process.load("RecoLocalTracker.SiStripRecHitConverter.OutOfTime_cff")
process.OutOfTime.TOBlateBP=0.071
process.OutOfTime.TIBlateBP=0.036

## new set of TOB Lorentz angle

from CondCore.DBCommon.CondDBSetup_cfi import *
process.stripLorentzAngle = cms.ESSource("PoolDBESSource",CondDBSetup,
                                         connect = cms.string('sqlite_file:/afs/cern.ch/user/b/benhoob/public/LorentzAngle/SiStripLorentzAngle_Deco.db'),
                                         toGet = cms.VPSet(cms.PSet(record = cms.string('SiStripLorentzAngleRcd'),
                                                                    tag = cms.string('SiStripLorentzAngle_Deco') ))
                                         )
process.es_prefer_stripLorentzAngle = cms.ESPrefer("PoolDBESSource", "stripLorentzAngle")

#############################################################################################################
### end of corrections to strip local reco
#############################################################################################################
### Tracker alignment
process.trackerAlignmentICHEP2010 = cms.ESSource("PoolDBESSource",CondDBSetup,
                                                 connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/PayLoads/BEAM10/TrackerAlignment_GR10_v1_offline/TOBCenteredObjectICHEP2010.db'),
                                                 toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),tag = cms.string('Alignments') ))
                                                 )
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignmentICHEP2010")


### Tracker APE
process.trackerAPEICHEP2010 = cms.ESSource("PoolDBESSource",CondDBSetup,
                                           connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/PayLoads/BEAM10/TrackerAlignmentErrors_GR10_v1_offline/APEforICHEP30umFPIX.db'),
                                           toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),tag = cms.string('AlignmentErrors') ))
                                           )
process.es_prefer_trackerAPE = cms.ESPrefer("PoolDBESSource", "trackerAPEICHEP2010")


########################## END NEW ALIGNMENT ######################

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
process.offlinePrimaryVertices.TkFilterParameters.minSiliconHits = 6
process.offlinePrimaryVertices.TkFilterParameters.maxD0Significance = 100
process.offlinePrimaryVertices.TkFilterParameters.minPixelHits = 1
process.offlinePrimaryVertices.TkClusParameters.zSeparation = 1


#######################
process.d0_phi_analyzer.BeamFitter.TrackCollection = 'TrackRefitter'
process.d0_phi_analyzer.BeamFitter.MinimumTotalLayers = 6
process.d0_phi_analyzer.BeamFitter.MinimumPixelLayers = -1
process.d0_phi_analyzer.BeamFitter.MaximumNormChi2 = 10
process.d0_phi_analyzer.BeamFitter.MinimumInputTracks = 2
process.d0_phi_analyzer.BeamFitter.MinimumPt = 1.0
process.d0_phi_analyzer.BeamFitter.MaximumImpactParameter = 1.0
process.d0_phi_analyzer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
process.d0_phi_analyzer.BeamFitter.InputBeamWidth = -1 # 0.0400
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
