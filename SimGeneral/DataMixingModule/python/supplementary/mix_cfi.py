import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup adding one zerobias event 
mix = cms.EDFilter("DataMixingModule",
    EErechitCollection = cms.InputTag("EcalRecHitProducer","EcalRecHitsEE"),
    Label = cms.string(''),
    # Si Pixels
    pixeldigiCollection = cms.InputTag("siPixelDigis"),
    ESRecHitCollectionDM = cms.string('EcalRecHitsES_DM'),
    CSCwiredigiCollection = cms.InputTag("muonCSCDigis"),
    maxBunch = cms.int32(1),
    HOrechitCollection = cms.InputTag("HORecHitCollection"),
    # Muons:
    DTdigiCollection = cms.InputTag("muonDTDigis"),
    CSCstripdigiCollection = cms.InputTag("muonCSCDigis"),
    input = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(25.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/mc/2007/6/12/CSA07-Minbias-1911/0000/007D90A2-7C1D-DC11-B793-001560AC4F10.root')
    ),
    HFRecHitCollectionDM = cms.string('HFRecHitCollection_DM'),
    HBHERecHitCollectionDM = cms.string('HBHERecHitCollection_DM'),
    # Si Strips
    SistripdigiCollection = cms.InputTag("siStripDigis"),
    DTDigiCollectionDM = cms.string('muonDTDigis_DM'),
    PixelDigiCollectionDM = cms.string('siPixelDigis_DM'),
    # Input Selectors/output tags
    # Ecal:	
    EBrechitCollection = cms.InputTag("EcalRecHitProducer","EcalRecHitsEB"),
    bunchspace = cms.int32(25),
    RPCdigiCollection = cms.InputTag("muonRPCDigis"),
    SiStripDigiCollectionDM = cms.string('siStripDigis_DM'),
    ZDCrechitCollection = cms.InputTag("ZDCRecHitCollection"),
    CSCWireDigiCollectionDM = cms.string('muonCSCDigis_DM'),
    HFrechitCollection = cms.InputTag("HFRecHitCollection"),
    RPCDigiCollectionDM = cms.string('muonRPCDigis_DM'),
    EERecHitCollectionDM = cms.string('EcalRecHitsEE_DM'),
    ESrechitCollection = cms.InputTag("ESRecHitProducer","EcalRecHitsES"),
    minBunch = cms.int32(-1),
    # Hcal:
    HBHErechitCollection = cms.InputTag("HBHERecHitCollection"),
    EBRecHitCollectionDM = cms.string('EcalRecHitsEB_DM'),
    ZDCRecHitCollectionDM = cms.string('ZDCRecHitCollection_DM'),
    CSCStripDigiCollectionDM = cms.string('muonCSCDigis_DM'),
    HORecHitCollectionDM = cms.string('HORecHitCollection_DM')
)


