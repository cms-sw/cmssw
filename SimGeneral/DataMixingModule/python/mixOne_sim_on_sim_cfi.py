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
    # Merge Pileup Info?
    MergePileupInfo = cms.bool(True),                         
    # Use digis?
    TrackerMergeType = cms.string("Digis"),
    EcalMergeType = cms.string('Digis'),  # set to "Digis" to merge digis
    HcalMergeType = cms.string('Digis'),
    HcalDigiMerge = cms.string('NotProd'), #use sim hits for signal                         
    #
    # Input Specifications:
    #

    PileupInfoInputTag = cms.InputTag("addPileupInfo"),
    BunchSpacingInputTag = cms.InputTag("addPileupInfo","bunchSpacing"),
    CFPlaybackInputTag = cms.InputTag("mix"),
    #
    SistripLabelSig = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
                   #
    pixeldigiCollectionSig = cms.InputTag("simSiPixelDigis"),
    #
    SiStripPileInputTag = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
                   #
    pixeldigiCollectionPile = cms.InputTag("simSiPixelDigis"),
                   #
    EBProducerSig = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EEProducerSig = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    ESProducerSig = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                   #
    HBHEProducerSig = cms.InputTag("hbhereco"),
    HOProducerSig = cms.InputTag("horeco"),                   
    HFProducerSig = cms.InputTag("hfreco"),
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

#    EBdigiCollectionSig = cms.InputTag("simEcalUnsuppressedDigis"),
#    EEdigiCollectionSig = cms.InputTag("simEcalUnsuppressedDigis"),
#    ESdigiCollectionSig = cms.InputTag("simEcalUnsuppressedDigis"),
                         
    EBdigiProducerSig = cms.InputTag("simEcalUnsuppressedDigis"),
    EEdigiProducerSig = cms.InputTag("simEcalUnsuppressedDigis"),
    ESdigiProducerSig = cms.InputTag("simEcalPreshowerDigis"),
    HBHEdigiCollectionSig  = cms.InputTag("simHcalUnsuppressedDigis"),
    HOdigiCollectionSig    = cms.InputTag("simHcalUnsuppressedDigis"),
    HFdigiCollectionSig    = cms.InputTag("simHcalUnsuppressedDigis"),
    ZDCdigiCollectionSig   = cms.InputTag("simHcalUnsuppressedDigis"),

    #
    EBPileInputTag = cms.InputTag("simEcalUnsuppressedDigis",""),
    EEPileInputTag = cms.InputTag("simEcalUnsuppressedDigis",""),
    ESPileInputTag = cms.InputTag("simEcalPreshowerDigis",""),
    HBHEPileInputTag = cms.InputTag("simHcalDigis"),
    HOPileInputTag   = cms.InputTag("simHcalDigis"),
    HFPileInputTag   = cms.InputTag("simHcalDigis"),
    ZDCPileInputTag  = cms.InputTag(""),

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
    DTPileInputTag        = cms.InputTag("simMuonDTDigis",""),
    RPCPileInputTag       = cms.InputTag("simMuonRPCDigis",""),
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


