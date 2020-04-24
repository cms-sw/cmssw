import FWCore.ParameterSet.Config as cms

mix = cms.EDFilter("DataMixingModule",
    input = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('fixed'),
        fileNames = cms.untracked.vstring('dcap://cmsdca.fnal.gov:24137/pnfs/fnal.gov/usr/cms/WAX/11/store/mc/CSA08/JetET30/GEN-SIM-RECO/CSA08_S156_v1/0002/000250F6-A72B-DD11-8904-00145E1D6204.root')
    ),
    # Mixing Module parameters
    Label = cms.string(''),
    maxBunch = cms.int32(0),
    bunchspace = cms.int32(25),
    minBunch = cms.int32(0),
    # Use digis?               
    EcalMergeType = cms.string('RecHits'),  # set to "Digis" to merge digis
    HcalMergeType = cms.string('RecHits'),
    #
    # Input Specifications:
    #
    SistripLabel = cms.InputTag("ZeroSuppressed"),
    SistripdigiCollection = cms.InputTag("simSiStripDigis"),
                   #
    pixeldigiCollection = cms.InputTag("simSiPixelDigis"),
                   #
    EBProducer = cms.InputTag("ecalRecHit"),
    EBrechitCollection = cms.InputTag("EcalRecHitsEB"),                   
    EEProducer = cms.InputTag("ecalRecHit"),                   
    EErechitCollection = cms.InputTag("EcalRecHitsEE"),
    ESProducer = cms.InputTag("ecalPreshowerRecHit"),
    ESrechitCollection = cms.InputTag("EcalRecHitsES"),                   
                   #
    HBHEProducer = cms.InputTag("hbhereco"),
    HBHErechitCollection = cms.InputTag("HBHERecHitCollection"),
    HOProducer = cms.InputTag("horeco"),                   
    HOrechitCollection = cms.InputTag("HORecHitCollection"),
    HFProducer = cms.InputTag("hfreco"),
    HFrechitCollection = cms.InputTag("HFRecHitCollection"),                   
    ZDCrechitCollection = cms.InputTag("ZDCRecHitCollection"),
    #
    # Calorimeter digis
    #
    EBdigiCollection = cms.InputTag("EBdigiCollection"),
    EEdigiCollection = cms.InputTag("EEdigiCollection"),
    ESdigiCollection = cms.InputTag("ESdigiCollection"),
    HBHEdigiCollection  = cms.InputTag("HBHEdigiCollection"),
    HOdigiCollection    = cms.InputTag("HOdigiCollection"),
    HFdigiCollection    = cms.InputTag("HFdigiCollection"),
    ZDCdigiCollection   = cms.InputTag("ZDCdigiCollection"),          
                   #
    CSCDigiTag = cms.InputTag("simMuonCSCDigis"),
    CSCwiredigiCollection = cms.InputTag("muonCSCWireDigi"),
    CSCstripdigiCollection = cms.InputTag("muonCSCStripDigi"),
    RPCDigiTag = cms.InputTag("simMuonRPCDigis"),                   
    RPCdigiCollection = cms.InputTag("simMuonRPCDigis"),
    DTDigiTag = cms.InputTag("simMuonDTDigis"),
    DTdigiCollection = cms.InputTag("simMuonDTDigis"),
                   #
    #
    #  Outputs
    #
    SiStripDigiCollectionDM = cms.string('siStripDigisDM'),
    PixelDigiCollectionDM = cms.string('siPixelDigisDM'),                   
    EBRecHitCollectionDM = cms.string('EcalRecHitsEBDM'),
    EERecHitCollectionDM = cms.string('EcalRecHitsEEDM'),                   
    ESRecHitCollectionDM = cms.string('EcalRecHitsESDM'),
    HBHERecHitCollectionDM = cms.string('HBHERecHitCollectionDM'),
    HFRecHitCollectionDM = cms.string('HFRecHitCollectionDM'),
    HORecHitCollectionDM = cms.string('HORecHitCollectionDM'),                   
    ZDCRecHitCollectionDM = cms.string('ZDCRecHitCollectionDM'),
    DTDigiCollectionDM = cms.string('muonDTDigisDM'),
    CSCWireDigiCollectionDM = cms.string('MuonCSCWireDigisDM'),
    CSCStripDigiCollectionDM = cms.string('MuonCSCStripDigisDM'),
    RPCDigiCollectionDM = cms.string('muonRPCDigisDM'),
    #
    #  Calorimeter Digis
    #               
    EBDigiCollectionDM   = cms.string('EBDigiCollectionDM'),
    EEDigiCollectionDM   = cms.string('EEDigiCollectionDM'),
    ESDigiCollectionDM   = cms.string('ESDigiCollectionDM'),
    HBHEDigiCollectionDM = cms.string('HBHEDigiCollectionDM'),
    HODigiCollectionDM   = cms.string('HODigiCollectionDM'),
    HFDigiCollectionDM   = cms.string('HFDigiCollectionDM'),
    ZDCDigiCollectionDM  = cms.string('ZDCDigiCollectionDM')
)


