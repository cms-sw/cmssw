import FWCore.ParameterSet.Config as cms

mix = cms.EDFilter("DataMixingModule",
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
    # Use digis?               
    EcalMergeType = cms.string('Digis'),  # set to "Digis" to merge digis
    HcalMergeType = cms.string('Digis'),
    #
    # Input Specifications:
    #
    SistripLabelSig = cms.InputTag("ZeroSuppressed"),
    SistripdigiCollectionSig = cms.InputTag("simSiStripDigis"),
                   #
    pixeldigiCollectionSig = cms.InputTag("simSiPixelDigis"),
    #
    SistripLabelPile = cms.InputTag("ZeroSuppressed"),
    SistripdigiCollectionPile = cms.InputTag("simSiStripDigis"),
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
    HBHEdigiCollectionSig  = cms.InputTag("simHcalDigis"),
    HOdigiCollectionSig    = cms.InputTag("simHcalDigis"),
    HFdigiCollectionSig    = cms.InputTag("simHcalDigis"),
    ZDCdigiCollectionSig   = cms.InputTag("ZDCdigiCollection"),          
    #
    EBdigiProducerPile = cms.InputTag("simEcalDigis"),
    EBdigiCollectionPile = cms.InputTag("ebDigis"),
    EEdigiProducerPile = cms.InputTag("simEcalDigis"),
    EEdigiCollectionPile = cms.InputTag("eeDigis"),
    ESdigiProducerPile = cms.InputTag("simEcalPreshowerDigis"),
    ESdigiCollectionPile = cms.InputTag(""),
    HBHEdigiCollectionPile  = cms.InputTag("simHcalDigis"),
    HOdigiCollectionPile    = cms.InputTag("simHcalDigis"),
    HFdigiCollectionPile    = cms.InputTag("simHcalDigis"),
    ZDCdigiCollectionPile   = cms.InputTag("ZDCdigiCollection"),          
#ESDataFramesSorted "simEcalPreshowerDigis" "" "HLT"
                   #HODataFramesSorted "simHcalDigis" "" "HLT"
                   #HFDataFramesSorted "simHcalDigis" "" "HLT"
                   #HBHEDataFramesSorted "simHcalDigis" "" "HLT"

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
    CSCDigiTagPile = cms.InputTag("simMuonCSCDigis"),
    CSCwiredigiCollectionPile = cms.InputTag("muonCSCWireDigi"),
    CSCstripdigiCollectionPile = cms.InputTag("muonCSCStripDigi"),
    RPCDigiTagPile = cms.InputTag("simMuonRPCDigis"),                   
    RPCdigiCollectionPile = cms.InputTag("simMuonRPCDigis"),
    DTDigiTagPile = cms.InputTag("simMuonDTDigis"),
    DTdigiCollectionPile = cms.InputTag("simMuonDTDigis"),
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


