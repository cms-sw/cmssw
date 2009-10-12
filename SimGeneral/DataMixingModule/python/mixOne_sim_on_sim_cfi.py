import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock

mixData = cms.EDFilter("DataMixingModule",
          hcalSimBlock,
    input = cms.SecSource("PoolRASource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(1.0)
        ),
        seed = cms.int32(1234567),
        type = cms.string('fixed'),
        fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0003/06C16DFA-9182-DD11-A4CC-000423D6CA6E.root')
    ),
    # Mixing Module parameters
    Label = cms.string(''),
    maxBunch = cms.int32(0),
    bunchspace = cms.int32(25),
    minBunch = cms.int32(0),
    checktof = cms.bool(False), 
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
    HBHErechitCollectionSig = cms.InputTag("HBHERecHitCollection"),
    HOProducerSig = cms.InputTag("horeco"),                   
    HOrechitCollectionSig = cms.InputTag("HORecHitCollection"),
    HFProducerSig = cms.InputTag("hfreco"),
    HFrechitCollectionSig = cms.InputTag("HFRecHitCollection"),                   
    ZDCrechitCollectionSig = cms.InputTag("ZDCRecHitCollection"),
    #
    #
    EBPileRecHitInputTag = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
    EEPileRecHitInputTag = cms.InputTag("ecalRecHit", "EcalRecHitsEE"),
    ESPileRecHitInputTag = cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES"),                  
    #
    HBHEPileRecHitInputTag = cms.InputTag("hbhereco", "HBHERecHitCollection"),
    HOPileRecHitInputTag = cms.InputTag("horeco", "HORecHitCollection"),                   
    HFPileRecHitInputTag = cms.InputTag("hfreco", "HFRecHitCollection"),
    ZDCPileRecHitInputTag = cms.InputTag("","ZDCRecHitCollection"),
    #
    # Calorimeter digis
    #
    EBdigiProducerSig = cms.InputTag("simEcalDigis"),
    EBdigiCollectionSig = cms.InputTag("ebDigis"),
    EEdigiProducerSig = cms.InputTag("simEcalDigis"),
    EEdigiCollectionSig = cms.InputTag("eeDigis"),
    ESdigiProducerSig = cms.InputTag("simEcalPreshowerDigis"),
    ESdigiCollectionSig = cms.InputTag(""),
    HBHEdigiCollectionSig  = cms.InputTag("simHcalDigis"),
    HOdigiCollectionSig    = cms.InputTag("simHcalDigis"),
    HFdigiCollectionSig    = cms.InputTag("simHcalDigis"),
    ZDCdigiCollectionSig   = cms.InputTag("ZDCdigiCollection"),          
    #
    EBPileInputTag = cms.InputTag("simEcalDigis","ebDigis"),
    EEPileInputTag = cms.InputTag("simEcalDigis","eeDigis"),
    ESPileInputTag = cms.InputTag("simEcalPreshowerDigis",""),
    HBHEPileInputTag = cms.InputTag("simHcalDigis"),                  
    HOPileInputTag   = cms.InputTag("simHcalDigis"),                  
    HFPileInputTag   = cms.InputTag("simHcalDigis"),                  
    ZDCPileInputTag  = cms.InputTag("ZDCdigiCollection"),          

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
    ESDigiCollectionDM   = cms.string('ESDigiCollectionDM'),
    HBHEDigiCollectionDM = cms.string('HBHEDigiCollectionDM'),
    HODigiCollectionDM   = cms.string('HODigiCollectionDM'),
    HFDigiCollectionDM   = cms.string('HFDigiCollectionDM'),
    ZDCDigiCollectionDM  = cms.string('ZDCDigiCollectionDM')
)


