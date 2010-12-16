import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock

mixData = cms.EDProducer("DataMixingModule",
          hcalSimBlock,
    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('fixed'),
        sequential = cms.untracked.bool(False), # set to true for sequential reading of pileup
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
    IsThisFastSim = cms.string('NO'),  # kludge for fast simulation flag...
    # Use digis?               
    EcalMergeType = cms.string('Digis'),  # set to "Digis" to merge digis
    HcalMergeType = cms.string('Digis'),
    HcalDigiMerge = cms.string('FullProd'), #use sim hits for signal
    #
    # Input Specifications:
    #
    SistripLabelSig = cms.InputTag("ZeroSuppressed"),
    SistripdigiCollectionSig = cms.InputTag("simSiStripDigis"),
                   #
    pixeldigiCollectionSig = cms.InputTag("simSiPixelDigis"),
    #
    SiStripPileInputTag = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
                   #
    pixeldigiCollectionPile = cms.InputTag("simSiPixelDigis"),
                   #
    EBProducerSig = cms.InputTag("ecalRecHit"),
    EBrechitCollectionSig = cms.InputTag("EcalRecHitsEB"),                   
    EEProducerSig = cms.InputTag("ecalRecHit"),                   
    EErechitCollectionSig = cms.InputTag("EcalRecHitsEE"),
    ESProducerSig = cms.InputTag("ecalPreshowerRecHit"),
    ESrechitCollectionSig = cms.InputTag("EcalRecHitsES"),                   
                   #
    HBHEProducerSig = cms.InputTag("hbhereco"),
    HBHErechitCollectionSig = cms.InputTag(""),
    HOProducerSig = cms.InputTag("horeco"),                   
    HOrechitCollectionSig = cms.InputTag(""),
    HFProducerSig = cms.InputTag("hfreco"),
    HFrechitCollectionSig = cms.InputTag(""),                   
    ZDCrechitCollectionSig = cms.InputTag("zdcreco"),
    #
    #
    EBPileRecHitInputTag = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    EEPileRecHitInputTag = cms.InputTag("ecalRecHit", "EcalRecHitsEE"),
    ESPileRecHitInputTag = cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES"),                  
    #
    HBHEPileRecHitInputTag = cms.InputTag("hbhereco", ""),
    HOPileRecHitInputTag = cms.InputTag("horeco", ""),                   
    HFPileRecHitInputTag = cms.InputTag("hfreco", ""),
    ZDCPileRecHitInputTag = cms.InputTag("zdcreco",""),
    #
    # Calorimeter digis
    #
EBdigiProducerSig = cms.InputTag("simEcalUnsuppressedDigis"),
    EBdigiCollectionSig = cms.InputTag(""),
    EEdigiProducerSig = cms.InputTag("simEcalUnsuppressedDigis"),
    EEdigiCollectionSig = cms.InputTag(""),
    ESdigiProducerSig = cms.InputTag("simEcalUnsuppressedDigis"),
    ESdigiCollectionSig = cms.InputTag(""),
    HBHEdigiCollectionSig  = cms.InputTag("simHcalUnsuppressedDigis"),
    HOdigiCollectionSig    = cms.InputTag("simHcalUnsuppressedDigis"),
    HFdigiCollectionSig    = cms.InputTag("simHcalUnsuppressedDigis"),
    ZDCdigiCollectionSig   = cms.InputTag("simHcalUnsuppressedDigis"),
    #
    EBPileInputTag = cms.InputTag("simEcalUnsuppressedDigis",""),
    EEPileInputTag = cms.InputTag("simEcalUnsuppressedDigis",""),
    ESPileInputTag = cms.InputTag("simEcalUnsuppressedDigis",""),
    HBHEPileInputTag = cms.InputTag("simHcalUnsuppressedDigis"),
    HOPileInputTag   = cms.InputTag("simHcalUnsuppressedDigis"),
    HFPileInputTag   = cms.InputTag("simHcalUnsuppressedDigis"),
    ZDCPileInputTag  = cms.InputTag("simHcalUnsuppressedDigis"),

    #  Signal
                   #
    CSCDigiTagSig = cms.InputTag("simMuonCSCDigis"),
    CSCwiredigiCollectionSig = cms.InputTag("MuonCSCWireDigi"),
    CSCstripdigiCollectionSig = cms.InputTag("MuonCSCStripDigi"),
    CSCCompdigiCollectionSig = cms.InputTag("MuonCSCComparatorDigi"),
    RPCDigiTagSig = cms.InputTag("simMuonRPCDigis"),                   
    RPCdigiCollectionSig = cms.InputTag("simMuonRPCDigis"),
    DTDigiTagSig = cms.InputTag("simMuonDTDigis"),
    DTdigiCollectionSig = cms.InputTag("simMuonDTDigis"),
    #  Pileup
                   #                   
    DTPileInputTag        = cms.InputTag("simMuonDTDigis","MuonDTDigis"),
    RPCPileInputTag       = cms.InputTag("simMuonRPCDigis","MuonRPCDigis"),
    CSCWirePileInputTag   = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    CSCStripPileInputTag  = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    CSCCompPileInputTag   = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
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


