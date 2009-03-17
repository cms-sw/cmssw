import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock

mix = cms.EDFilter("DataMixingModule",
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
    SistripLabelPile = cms.InputTag("ZeroSuppressed"),
    SistripdigiCollectionPile = cms.InputTag("siStripDigis"),
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
    EBProducerPile = cms.InputTag("ecalRecHit"),
    EBrechitCollectionPile = cms.InputTag("EcalRecHitsEB"),                   
    EEProducerPile = cms.InputTag("ecalRecHit"),                   
    EErechitCollectionPile = cms.InputTag("EcalRecHitsEE"),
    ESProducerPile = cms.InputTag("ecalPreshowerRecHit"),
    ESrechitCollectionPile = cms.InputTag("EcalRecHitsES"),                   
                   #
    HBHEProducerPile = cms.InputTag("hbhereco"),
    HBHErechitCollectionPile = cms.InputTag("HBHERecHitCollection"),
    HOProducerPile = cms.InputTag("horeco"),                   
    HOrechitCollectionPile = cms.InputTag("HORecHitCollection"),
    HFProducerPile = cms.InputTag("hfreco"),
    HFrechitCollectionPile = cms.InputTag("HFRecHitCollection"),                   
    ZDCrechitCollectionPile = cms.InputTag("ZDCRecHitCollection"),
    #
    # Calorimeter digis
    #
    EBdigiProducerSig = cms.InputTag("simEcalDigis"),
    EBdigiCollectionSig = cms.InputTag("ebDigis"),
    EEdigiProducerSig = cms.InputTag("simEcalDigis"),
    EEdigiCollectionSig = cms.InputTag("eeDigis"),
    ESdigiProducerSig = cms.InputTag("simEcalPreshowerDigis"),
    ESdigiCollectionSig = cms.InputTag(""),
    # 
    EBdigiCollectionPile = cms.InputTag("ebDigis"),
    EEdigiCollectionPile = cms.InputTag("eeDigis"),
    ESdigiCollectionPile = cms.InputTag(""),
    EBdigiProducerPile = cms.InputTag("ecalDigis"),
    EEdigiProducerPile = cms.InputTag("ecalDigis"),
    ESdigiProducerPile = cms.InputTag("ecalPreshowerDigis"),
    #                   
    #
    HBHEdigiCollectionSig  = cms.InputTag("simHcalDigis"),
    HOdigiCollectionSig    = cms.InputTag("simHcalDigis"),
    HFdigiCollectionSig    = cms.InputTag("simHcalDigis"),
    ZDCdigiCollectionSig   = cms.InputTag("ZDCdigiCollection"),          
    #
    HBHEdigiCollectionPile  = cms.InputTag("hcalDigis"),
    HOdigiCollectionPile    = cms.InputTag("hcalDigis"),
    HFdigiCollectionPile    = cms.InputTag("hcalDigis"),
    ZDCdigiCollectionPile   = cms.InputTag("ZDCdigiCollection"),          
    #  Signal
                   #
    CSCDigiTagSig = cms.InputTag("simMuonCSCDigis"),
    CSCwiredigiCollectionSig = cms.InputTag("muonCSCWireDigi"),
    CSCstripdigiCollectionSig = cms.InputTag("muonCSCStripDigi"),
    RPCDigiTagSig = cms.InputTag("simMuonRPCDigis"),                   
    RPCdigiCollectionSig = cms.InputTag("simMuonRPCDigis"),
    DTDigiTagSig = cms.InputTag("simMuonDTDigis"),
    DTdigiCollectionSig = cms.InputTag("simMuonDTDigis"),
    #  Pileup
                   #                   
    CSCDigiTagPile = cms.InputTag("muonCSCDigis"),
    CSCwiredigiCollectionPile = cms.InputTag("muonCSCWireDigi"),
    CSCstripdigiCollectionPile = cms.InputTag("muonCSCStripDigi"),
    RPCDigiTagPile = cms.InputTag("muonRPCDigis"),                   
    RPCdigiCollectionPile = cms.InputTag("MuonRPCDigis"),
    DTDigiTagPile = cms.InputTag("muonDTDigis"),
    DTdigiCollectionPile = cms.InputTag("MuonDTDigis"),
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


