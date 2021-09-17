import FWCore.ParameterSet.Config as cms

process = cms.Process("MULTITRACKVALIDATORGENPS")

# message logger
process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.TwoTrackMinimumDistanceLineLine = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1),
    limit = cms.untracked.int32( 10 )
)
process.MessageLogger.suppressWarning = cms.untracked.vstring('multiTrackValidatorGenPs')

# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

#readFiles.extend( [
#    '/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0A962BB7-7290-E111-ABF6-003048FFD720.root',
#    '/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0A8181A3-B890-E111-BEC7-003048FFCB84.root',
#    '/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0A7E802E-6E90-E111-B850-0018F3D0962E.root',
#    '/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0A756CC1-8E90-E111-AD35-002618943930.root',
#    '/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0A55D5A7-8290-E111-9779-002618943914.root',
#    '/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0A430829-B98F-E111-891F-0026189438AA.root',
#    '/store/mc/Summer12/TTJets_TuneZ2star_8TeV-madgraph-tauola/AODSIM/PU_S7_START52_V9-v1/0000/0A0AB31B-A490-E111-A735-001A92810AD8.root'
#    ] );

#readFiles.extend( [
#    '/store/relval/CMSSW_6_1_0_pre5-START61_V4/RelValTTbar/GEN-SIM-RECO/v1/00000/26FB671D-0D27-E211-96DC-002618943833.root',
#    '/store/relval/CMSSW_6_1_0_pre5-START61_V4/RelValTTbar/GEN-SIM-RECO/v1/00000/1EBD597F-0927-E211-A03A-002618943849.root' ] );

readFiles.extend( [
    '/store/relval/CMSSW_6_1_0_pre5-PU_START61_V4/RelValTTbar/GEN-SIM-RECO/v2/00000/FA718DED-4C29-E211-B337-002354EF3BDA.root',
    '/store/relval/CMSSW_6_1_0_pre5-PU_START61_V4/RelValTTbar/GEN-SIM-RECO/v2/00000/D6FA7621-7F29-E211-B8D1-00248C55CC97.root',
    '/store/relval/CMSSW_6_1_0_pre5-PU_START61_V4/RelValTTbar/GEN-SIM-RECO/v2/00000/D44FCB0B-3429-E211-8C07-003048FFCB6A.root',
    '/store/relval/CMSSW_6_1_0_pre5-PU_START61_V4/RelValTTbar/GEN-SIM-RECO/v2/00000/C6EC8844-3129-E211-840F-002618943849.root',
    '/store/relval/CMSSW_6_1_0_pre5-PU_START61_V4/RelValTTbar/GEN-SIM-RECO/v2/00000/6E4C6838-6029-E211-AF06-003048FFD756.root',
    '/store/relval/CMSSW_6_1_0_pre5-PU_START61_V4/RelValTTbar/GEN-SIM-RECO/v2/00000/60314571-3229-E211-841D-003048678BAC.root',
    '/store/relval/CMSSW_6_1_0_pre5-PU_START61_V4/RelValTTbar/GEN-SIM-RECO/v2/00000/309E8B5D-3629-E211-9FEF-0026189438E0.root'] );

process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(250) )

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START61_V4::All'

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

### validation-specific includes
process.load("SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi")
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi")
process.load("Validation.RecoTrack.MultiTrackValidatorGenPs_cfi")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("Validation.Configuration.postValidation_cff")
process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')

process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")
process.trackAssociatorByChi2.chi2cut = cms.double(500.0)
process.trackAssociatorByPull = process.trackAssociatorByChi2.clone(chi2cut = 50.0,onlyDiagonal = True)


########### configuration MultiTrackValidatorGenPs ########
process.multiTrackValidatorGenPs.outputFile = 'multitrackvalidatorgenps.root'
process.multiTrackValidatorGenPs.associators = ['trackAssociatorByChi2','trackAssociatorByPull']
process.multiTrackValidatorGenPs.UseAssociators = cms.bool(True)
process.MTVHistoProducerAlgoForTrackerBlock.maxPt = cms.double(1100)

process.load("Validation.RecoTrack.cuts_cff")
#process.cutsRecoTracks.ptMin    = cms.double(0.5)
#process.cutsRecoTracks.minRapidity  = cms.int32(-1.0)
#process.cutsRecoTracks.maxRapidity  = cms.int32(1.0)
process.cutsRecoTracks.quality = cms.vstring('highPurity')
#process.cutsRecoTracks.min3DHit = cms.int32(3)
#process.cutsRecoTracks.minPixHit = cms.int32(0)
#process.cutsRecoTracks.algorithm = cms.vstring('tobTecStep')
#process.cutsRecoTracks.maxChi2 = 10
#process.cutsRecoTracks.minHit   = cms.int32(10)
#process.cutsRecoTracks.src = cms.InputTag("TrackRefitter")

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.selectedVertices = cms.EDFilter("VertexSelector",
    src = cms.InputTag('offlinePrimaryVertices'),
    cut = cms.string("isValid & ndof >= 4 & chi2 > 0 & tracksSize > 0 & abs(z) < 24 & abs(position.Rho) < 2."),
    filter = cms.bool(False)                                          
)

process.selectedFirstPrimaryVertex = cms.EDFilter("PATSingleVertexSelector",
    mode = cms.string('firstVertex'),
    vertices = cms.InputTag('selectedVertices'),
    filter = cms.bool(False)                                                    
)

process.trackWithVertexSelector = cms.EDProducer("TrackWithVertexSelector",
    # -- module configuration --
    src = cms.InputTag('generalTracks'),
    quality = cms.string("highPurity"),
    useVtx = cms.bool(True),
    vertexTag = cms.InputTag('selectedFirstPrimaryVertex'),
    nVertices = cms.uint32(1),
    vtxFallback = cms.bool(False),
    copyExtras = cms.untracked.bool(False),
    copyTrajectories = cms.untracked.bool(False),
    # --------------------------
    # -- these are the vertex compatibility cuts --
    zetaVtx = cms.double(0.2),
    rhoVtx = cms.double(0.1),
    # ---------------------------------------------
    # -- dummy selection on tracks --
    etaMin = cms.double(0.0),
    etaMax = cms.double(5.0),
    ptMin = cms.double(0.00001),
    ptMax = cms.double(999999.),
    d0Max = cms.double(999999.),
    dzMax = cms.double(999999.),
    normalizedChi2 = cms.double(999999.),
    numberOfValidHits = cms.uint32(0),
    numberOfLostHits = cms.uint32(999),
    numberOfValidPixelHits = cms.uint32(0),
    ptErrorCut = cms.double(999999.)
    # ------------------------------                                       
)


#process.multiTrackValidatorGenPs.label = ['cutsRecoTracks']
#process.multiTrackValidatorGenPs.label = ['generalTracks']
process.multiTrackValidatorGenPs.label = ['trackWithVertexSelector']

process.tracking = cms.Sequence(
    process.siPixelRecHits
    * process.siStripMatchedRecHits
    * process.clusterSummaryProducer
    * process.trackingGlobalReco
)

process.refit = cms.Sequence(
    process.TrackRefitter
)

process.validation = cms.Sequence(
    #process.cutsRecoTracks *
    process.selectedVertices*process.selectedFirstPrimaryVertex*process.trackWithVertexSelector *
    process.trackAssociatorByChi2 *
    process.trackAssociatorByPull *
    process.multiTrackValidatorGenPs
)

# paths
process.p = cms.Path(
    #process.tracking *
    #process.refit *
    process.validation
)
process.schedule = cms.Schedule(
      process.p
)


