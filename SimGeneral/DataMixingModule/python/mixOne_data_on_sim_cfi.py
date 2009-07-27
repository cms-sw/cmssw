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
#        fileNames = cms.untracked.vstring('dcap://cmsdca.fnal.gov:24137/pnfs/fnal.gov/usr/cms/WAX/11/store/mc/CSA08/JetET30/GEN-SIM-RECO/CSA08_S156_v1/0002/000250F6-A72B-DD11-8904-00145E1D6204.root')
#        fileNames = cms.untracked.vstring('file:/uscms/home/mikeh/work/CMSSW_3_1_0_pre1/src/SimGeneral/DataMixingModule/python/promptrecoCosmicsDigis.root')
         fileNames = cms.untracked.vstring('file:/uscms_data/d1/mikeh/promptRecoHCalNoise.root')
    ),
    # Mixing Module parameters
    Label = cms.string(''),
    maxBunch = cms.int32(0),
    bunchspace = cms.int32(25),
    minBunch = cms.int32(0),
    checktof = cms.bool(False),
    #
    IsThisFastSim = cms.string('NO'),  # kludge for fast simulation flag...
    # Use digis?               
    EcalMergeType = cms.string('Digis'),  # set to "Digis" to merge digis
    HcalMergeType = cms.string('Digis'),
    HcalDigiMerge = cms.string('FullProd'),
    #
    # Input Specifications:
    #
    SistripLabelSig = cms.InputTag("ZeroSuppressed"),
    SistripdigiCollectionSig = cms.InputTag("simSiStripDigis"),
                   #
    pixeldigiCollectionSig = cms.InputTag("simSiPixelDigis"),
    #
    SiStripPileInputTag = cms.InputTag("siStripDigis","ZeroSuppressed"),
                   #
    pixeldigiCollectionPile = cms.InputTag("siPixelDigis"),
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
    #EBdigiProducerSig = cms.InputTag("simUnsupressedEcalDigis"),
    EBdigiCollectionSig = cms.InputTag("ebDigis"),
    EEdigiProducerSig = cms.InputTag("simEcalDigis"),
    #EEdigiProducerSig = cms.InputTag("simUnsupressedEcalDigis"),
    EEdigiCollectionSig = cms.InputTag("eeDigis"),
    ESdigiProducerSig = cms.InputTag("simEcalPreshowerDigis"),
    ESdigiCollectionSig = cms.InputTag(""),
    # 
    HBHEdigiCollectionSig  = cms.InputTag("simHcalDigis"),
    HOdigiCollectionSig    = cms.InputTag("simHcalDigis"),
    HFdigiCollectionSig    = cms.InputTag("simHcalDigis"),
    ZDCdigiCollectionSig   = cms.InputTag("ZDCdigiCollection"),          
    #
    EBPileInputTag = cms.InputTag("ecalDigis","ebDigis"),
    EEPileInputTag = cms.InputTag("ecalDigis","eeDigis"),
    ESPileInputTag = cms.InputTag("ecalPreshowerDigis",""),
    HBHEPileInputTag = cms.InputTag("hcalDigis"),                  
    HOPileInputTag   = cms.InputTag("hcalDigis"),                  
    HFPileInputTag   = cms.InputTag("hcalDigis"),                  
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
    DTPileInputTag        = cms.InputTag("muonDTDigis","MuonDTDigis"),
    RPCPileInputTag       = cms.InputTag("muonRPCDigis","MuonRPCDigis"),
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
    ESDigiCollectionDM   = cms.string('ESDigiCollectionDM'),
    HBHEDigiCollectionDM = cms.string('HBHEDigiCollectionDM'),
    HODigiCollectionDM   = cms.string('HODigiCollectionDM'),
    HFDigiCollectionDM   = cms.string('HFDigiCollectionDM'),
    ZDCDigiCollectionDM  = cms.string('ZDCDigiCollectionDM')
)


