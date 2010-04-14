import FWCore.ParameterSet.Config as cms

process = cms.Process("EXOHSCPAnalysis")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#number of Events to be skimmed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START3X_V26::All'
#process.GlobalTag.globaltag = 'GR_R_35X_V6::All'

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEFC70B6-F53D-DF11-B57E-003048679150.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FED8673E-F53D-DF11-9E58-0026189437EB.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEBF7874-EF3D-DF11-910D-002354EF3BDF.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEA8ECD8-F13D-DF11-8EBD-00304867BFAE.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE838E9F-F43D-DF11-BEBA-00261894393B.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE7D760E-F43D-DF11-878A-00304867BED8.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE2D63AD-F43D-DF11-B2B8-00261894395C.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC95A7F1-F13D-DF11-8C91-003048678C9A.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC5F5CA1-F53D-DF11-AFEE-002618FDA211.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC140D7E-F43D-DF11-B6C2-0026189437ED.root',
   )
)

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.dedxHarm2 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("TrackRefitter"),
    trajectoryTrackAssociation = cms.InputTag("TrackRefitter"),

    estimator      = cms.string('generic'),
    exponent       = cms.double(-2.0),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(True),
    MeVperADCStrip = cms.double(3.61e-06*250),
    MeVperADCPixel = cms.double(3.61e-06),

    MisCalib_Mean      = cms.untracked.double(1.0),
    MisCalib_Sigma     = cms.untracked.double(0.00),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
)

process.dedxNPHarm2             = process.dedxHarm2.clone()
process.dedxNPHarm2.UsePixel    = cms.bool(False)


#ALL THIS IS NEEDED BY ECAL BETA CALCULATOR (TrackAssociator)
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
from TrackingTools.TrackAssociator.default_cfi import * 
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")
process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi")

process.HSCParticleProducer = cms.EDProducer("HSCParticleProducer",
   TrackAssociatorParameterBlock,

   useBetaFromTk      = cms.bool(True),
   useBetaFromMuon    = cms.bool(True),
   useBetaFromRpc     = cms.bool(False),
   useBetaFromEcal    = cms.bool(False),

   tracks             = cms.InputTag("TrackRefitter"),
   muons              = cms.InputTag("muons"),
   dEdXEstimator      = cms.InputTag("dedxNPHarm2"),
   muontiming         = cms.InputTag("muons:combined"),

   minTkP             = cms.double(2),
   maxTkChi2          = cms.double(5),
   minTkHits          = cms.uint32(9),

   minMuP             = cms.double(2),

   minDR              = cms.double(0.1),
   maxInvPtDiff       = cms.double(0.005),

   minTkdEdx          = cms.double(2.0),
   maxMuBeta          = cms.double(0.9),

)


process.HSCParticleSelector = cms.EDProducer("HSCParticleSelector",
   source             = cms.InputTag("HSCParticleProducer"),

   minTkdEdx          = cms.double(4.0),
   maxMuBeta          = cms.double(0.9),
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('tfile_out.root')
)

process.p = cms.Path(process.offlineBeamSpot + process.TrackRefitter + process.dedxHarm2 + process.dedxNPHarm2 + process.HSCParticleProducer + process.HSCParticleSelector)

process.OUT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("drop *", "keep *_*_*_EXOHSCPAnalysis"),
    fileName = cms.untracked.string('out.root')
)

process.endPath = cms.EndPath(process.OUT)




