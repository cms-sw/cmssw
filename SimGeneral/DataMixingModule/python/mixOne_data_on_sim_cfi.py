import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock

# temporary fixes for sample size mismatch in HF (Data vs MC).

hcalSimBlock.hf1.readoutFrameSize = 10
hcalSimBlock.hf2.readoutFrameSize = 10
hcalSimBlock.hf1.binOfMaximum = 5
hcalSimBlock.hf2.binOfMaximum = 5

##################################


mixData = cms.EDProducer("DataMixingModule",
                   hcalSimBlock,
    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('fixed'),
        sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
            'file:DMPreProcess_RAW2DIGI.root'
        )
    ),
    # Mixing Module parameters
    Label = cms.string(''),
    maxBunch = cms.int32(0),
    bunchspace = cms.int32(25),
    minBunch = cms.int32(0),
    #
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),
    # Use digis? 
    TrackerMergeType = cms.string("Digis"),
    EcalMergeType = cms.string('Digis'),  # set to "Digis" to merge digis
    HcalMergeType = cms.string('Digis'),
    HcalDigiMerge = cms.string('FullProd'),
    #
    # Input Specifications:
    #
    SiStripRawDigiSource = cms.string("NONE"), # rawdigi+digi->rawdigi (specify 'SIGNAL' or 'PILEUP')
    SiStripRawInputTag = cms.InputTag("siStripDigis","VirginRaw"),               
    #
    SistripLabelSig = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
                   #
    pixeldigiCollectionSig = cms.InputTag("simSiPixelDigis"),
    #
    SiStripPileInputTag = cms.InputTag("siStripDigis","ZeroSuppressed"),
                   #
    pixeldigiCollectionPile = cms.InputTag("siPixelDigis"),
                   #
    EBProducerSig = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EEProducerSig = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    ESProducerSig = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                   #
    HBHEProducerSig = cms.InputTag("hbhereco"),
    HOProducerSig = cms.InputTag("horeco"),
    HFProducerSig = cms.InputTag("hfreco"),
    ZDCrechitCollectionSig = cms.InputTag("wrongTag"),
#                         ZDCrechitCollectionSig = cms.InputTag("zdcreco"),
    #
    EBPileRecHitInputTag = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    EEPileRecHitInputTag = cms.InputTag("ecalRecHit", "EcalRecHitsEE"),
    ESPileRecHitInputTag = cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES"),                  
    #
    HBHEPileRecHitInputTag = cms.InputTag("hbhereco", ""),
    HOPileRecHitInputTag = cms.InputTag("horeco", ""),                   
    HFPileRecHitInputTag = cms.InputTag("hfreco", ""),
    ZDCPileRecHitInputTag = cms.InputTag("wrongTag",""),
#                         ZDCPileRecHitInputTag = cms.InputTag("zdcreco",""),
    #
    # Calorimeter digis
    #
    EBdigiCollectionSig = cms.InputTag("simEcalUnsuppressedDigis"),
    EEdigiCollectionSig = cms.InputTag("simEcalUnsuppressedDigis"),
    ESdigiCollectionSig = cms.InputTag("simEcalUnsuppressedDigis"),
                         
    EBdigiProducerSig = cms.InputTag("simEcalUnsuppressedDigis"),
    EEdigiProducerSig = cms.InputTag("simEcalUnsuppressedDigis"),
    ESdigiProducerSig = cms.InputTag("simEcalUnsuppressedDigis"),
    HBHEdigiCollectionSig  = cms.InputTag("simHcalUnsuppressedDigis"),
    HOdigiCollectionSig    = cms.InputTag("simHcalUnsuppressedDigis"),
    HFdigiCollectionSig    = cms.InputTag("simHcalUnsuppressedDigis"),
    ZDCdigiCollectionSig   = cms.InputTag("simHcalUnsuppressedDigis"),

    # Sim Level (for Prod mode) ?
    # hitsProducer=cms.string("g4SimHits")
    #
    EBPileInputTag = cms.InputTag("ecalDigis","ebDigis"),
    EEPileInputTag = cms.InputTag("ecalDigis","eeDigis"),
    ESPileInputTag = cms.InputTag("ecalPreshowerDigis",""),
    HBHEPileInputTag = cms.InputTag("hcalDigis"),                  
    HOPileInputTag   = cms.InputTag("hcalDigis"),                  
    HFPileInputTag   = cms.InputTag("hcalDigis"),                  
    ZDCPileInputTag  = cms.InputTag("hcalDigis"),
    #  Signal
                   #
    CSCwiredigiCollectionSig = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    CSCstripdigiCollectionSig = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    CSCCompdigiCollectionSig = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    RPCDigiTagSig = cms.InputTag("simMuonRPCDigis"),                   
    RPCdigiCollectionSig = cms.InputTag("simMuonRPCDigis"),
    DTDigiTagSig = cms.InputTag("simMuonDTDigis"),
    DTdigiCollectionSig = cms.InputTag("simMuonDTDigis"),
    #  Pileup
                   #                   
    DTPileInputTag        = cms.InputTag("muonDTDigis"),
    RPCPileInputTag       = cms.InputTag("muonRPCDigis"),
    CSCWirePileInputTag   = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    CSCStripPileInputTag  = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    CSCCompPileInputTag   = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
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
    CSCComparatorDigiCollectionDM = cms.string('MuonCSCComparatorDigisDM'),
    RPCDigiCollectionDM = cms.string('muonRPCDigisDM'),
    #
    #  Calorimeter Digis
    #               
    EBDigiCollectionDM   = cms.string('EBDigiCollectionDM'),
    EEDigiCollectionDM   = cms.string('EEDigiCollectionDM'),
    ESDigiCollectionDM   = cms.string(''),
    HBHEDigiCollectionDM = cms.string(''),
    HODigiCollectionDM   = cms.string(''),
    HFDigiCollectionDM   = cms.string(''),
    ZDCDigiCollectionDM  = cms.string('')
)


