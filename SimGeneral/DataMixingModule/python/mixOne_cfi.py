import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup adding one zerobias event 
mix = cms.EDFilter("DataMixingModule",
    SistripLabel = cms.InputTag("ZeroSuppressed"),
    EErechitCollection = cms.InputTag("EcalRecHitsEE"),
    CSCDigiTag = cms.InputTag("muonCSCDigis"),
    Label = cms.string(''),
    # Si Pixels
    pixeldigiCollection = cms.InputTag("siPixelDigis"),
    ESRecHitCollectionDM = cms.string('EcalRecHitsESDM'),
    EEProducer = cms.InputTag("ecalRecHit"),
    CSCwiredigiCollection = cms.InputTag("MuonCSCWireDigi"),
    # Muons:
    DTdigiCollection = cms.InputTag("muonDTDigis"),
    DTDigiTag = cms.InputTag("muonDTDigis"),
    maxBunch = cms.int32(0),
    HOrechitCollection = cms.InputTag("HORecHitCollection"),
    # Hcal:
    HBHEProducer = cms.InputTag("hbhereco"),
    CSCstripdigiCollection = cms.InputTag("MuonCSCStripDigi"),
    input = cms.SecSource("PoolRASource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('fixed'),
        fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/mikeh/cms/CMSSW_2_0_0_pre2/src/Configuration/Examples/data/PhysValWDigi-DiElectron-2Ene30.root')
    ),
    HFRecHitCollectionDM = cms.string('HFRecHitCollectionDM'),
    HBHERecHitCollectionDM = cms.string('HBHERecHitCollectionDM'),
    # Si Strips
    SistripdigiCollection = cms.InputTag("siStripDigis"),
    # Input Selectors/output tags
    # Ecal:	
    #    	InputTag EBrechitCollection = EcalRecHitProducer:EcalRecHitsEB
    EBProducer = cms.InputTag("ecalRecHit"),
    DTDigiCollectionDM = cms.string('muonDTDigisDM'),
    HOProducer = cms.InputTag("horeco"),
    PixelDigiCollectionDM = cms.string('siPixelDigisDM'),
    EBrechitCollection = cms.InputTag("EcalRecHitsEB"),
    bunchspace = cms.int32(25),
    RPCdigiCollection = cms.InputTag("muonRPCDigis"),
    SiStripDigiCollectionDM = cms.string('siStripDigisDM'),
    RPCDigiTag = cms.InputTag("muonRPCDigis"),
    HFProducer = cms.InputTag("hfreco"),
    ZDCrechitCollection = cms.InputTag("ZDCRecHitCollection"),
    ESProducer = cms.InputTag("ecalPreshowerRecHit"),
    CSCWireDigiCollectionDM = cms.string('MuonCSCWireDigisDM'),
    HFrechitCollection = cms.InputTag("HFRecHitCollection"),
    RPCDigiCollectionDM = cms.string('muonRPCDigisDM'),
    EERecHitCollectionDM = cms.string('EcalRecHitsEEDM'),
    ESrechitCollection = cms.InputTag("EcalRecHitsES"),
    minBunch = cms.int32(0),
    HBHErechitCollection = cms.InputTag("HBHERecHitCollection"),
    EBRecHitCollectionDM = cms.string('EcalRecHitsEBDM'),
    ZDCRecHitCollectionDM = cms.string('ZDCRecHitCollectionDM'),
    CSCStripDigiCollectionDM = cms.string('MuonCSCStripDigisDM'),
    HORecHitCollectionDM = cms.string('HORecHitCollectionDM')
)


