import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock
import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
import EventFilter.ESRawToDigi.esRawToDigi_cfi
import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
import EventFilter.DTRawToDigi.dtunpacker_cfi
import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
import EventFilter.CSCRawToDigi.cscUnpacker_cfi
import EventFilter.SiStripRawToDigi.SiStripDigis_cfi
import EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi

# content from Configuration/StandardSequences/DigiToRaw_cff.py

ecalDigis = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone()

ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()

hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()

muonCSCDigis = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone()

muonDTDigis = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone()

muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()

#castorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone( FEDs = cms.untracked.vint32(690,691,692) )

siStripDigis = EventFilter.SiStripRawToDigi.SiStripDigis_cfi.siStripDigis.clone()

siPixelDigis = EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi.siPixelDigis.clone()

siPixelDigis.InputLabel = 'rawDataCollector'
ecalDigis.InputLabel = 'rawDataCollector'
ecalPreshowerDigis.sourceTag = 'rawDataCollector'
hcalDigis.InputLabel = 'rawDataCollector'
muonCSCDigis.InputObjects = 'rawDataCollector'
muonDTDigis.inputLabel = 'rawDataCollector'
muonRPCDigis.InputLabel = 'rawDataCollector'
#castorDigis.InputLabel = 'rawDataCollector'


mixData = cms.EDProducer("DataMixingModule",
          hcalSimBlock,
    input = cms.SecSource("PoolSource",
        producers = cms.VPSet(cms.convertToVPSet(
                                             ecalDigis = ecalDigis,
                                             ecalPreshowerDigis = ecalPreshowerDigis,
                                             hcalDigis = hcalDigis,
                                             muonDTDigis = muonDTDigis,
                                             muonRPCDigis = muonRPCDigis,
                                             muonCSCDigis = muonCSCDigis,
                                             siStripDigis = siStripDigis,
                                             siPixelDigis = siPixelDigis,
                             )),
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
    # Merge Pileup Info?
    MergePileupInfo = cms.bool(True),                         
    # Use digis?               
    EcalMergeType = cms.string('Digis'),  # set to "Digis" to merge digis
    HcalMergeType = cms.string('Digis'),
    HcalDigiMerge = cms.string('NotProd'), #use sim hits for signal
    #
    # Input Specifications:
    #

    PileupInfoInputTag = cms.InputTag("addPileupInfo"),
    CFPlaybackInputTag = cms.InputTag("mix"),
    #
    SistripLabelSig = cms.InputTag("simSiStripDigis","ZeroSuppressed"),
                   #
    pixeldigiCollectionSig = cms.InputTag("simSiPixelDigis"),
    #
    SiStripPileInputTag = cms.InputTag("siStripDigis","ZeroSuppressed","@MIXING"),
                   #
    pixeldigiCollectionPile = cms.InputTag("siPixelDigis","","@MIXING"),
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
    EBPileInputTag = cms.InputTag("ecalEBunpacker","ebDigis","@MIXING"),
    EEPileInputTag = cms.InputTag("ecalEBunpacker","eeDigis","@MIXING"),
    ESPileInputTag = cms.InputTag("esRawToDigi","","@MIXING"),
    HBHEPileInputTag = cms.InputTag("hcalDigis","","@MIXING"),
    HOPileInputTag   = cms.InputTag("hcalDigis","","@MIXING"),
    HFPileInputTag   = cms.InputTag("hcalDigis","","@MIXING"),
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
    DTPileInputTag        = cms.InputTag("muonDTDigis","","@MIXING"),
    RPCPileInputTag       = cms.InputTag("rpcunpacker","","@MIXING"),
    CSCWirePileInputTag   = cms.InputTag("muonCSCDigis","MuonCSCWireDigi","@MIXING"),
    CSCStripPileInputTag  = cms.InputTag("muonCSCDigis","MuonCSCStripDigi","@MIXING"),
    CSCCompPileInputTag   = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi","@MIXING"),
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


