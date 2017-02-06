# This is the PreMixing config for the DataMixer.  Not only does it do a RawToDigi conversion
# to the secondary input source, it also holds its own instances of an EcalDigiProducer and
# an HcalDigitizer.  It also replicates the noise adding functions in the SiStripDigitizer.
#


import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock
from SimGeneral.MixingModule.SiStripSimParameters_cfi import SiStripSimBlock
from SimGeneral.MixingModule.SiPixelSimParameters_cfi import SiPixelSimBlock
from SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff import *
from SimCalorimetry.EcalSimProducers.apdSimParameters_cff import *
from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.ecalElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.esElectronicsSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *
from SimCalorimetry.EcalSimProducers.ecalCosmicsSim_cff import *

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

#muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()
#castorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone( FEDs = cms.untracked.vint32(690,691,692) )

siStripDigis = EventFilter.SiStripRawToDigi.SiStripDigis_cfi.siStripDigis.clone()

siPixelDigis = EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi.siPixelDigis.clone()

siPixelDigis.InputLabel = 'rawDataCollector'
ecalDigis.InputLabel = 'rawDataCollector'
ecalPreshowerDigis.sourceTag = 'rawDataCollector'
hcalDigis.InputLabel = 'rawDataCollector'
muonCSCDigis.InputObjects = 'rawDataCollector'
muonDTDigis.inputLabel = 'rawDataCollector'
#muonRPCDigis.InputLabel = 'rawDataCollector'
#castorDigis.InputLabel = 'rawDataCollector'

hcalDigis.FilterDataQuality = cms.bool(False)
hcalSimBlock.HcalPreMixStage2 = cms.bool(True)

mixData = cms.EDProducer("DataMixingModule",
          hcalSimBlock,
          SiStripSimBlock,
          SiPixelSimBlock,
          ecal_digi_parameters,
          apd_sim_parameters,
          ecal_electronics_sim,
          ecal_cosmics_sim,
          ecal_sim_parameter_map,
          ecal_notCont_sim,
          es_electronics_sim,
    input = cms.SecSource("EmbeddedRootSource",
        producers = cms.VPSet(cms.convertToVPSet(
                                             ecalDigis = ecalDigis,
                                             ecalPreshowerDigis = ecalPreshowerDigis,
                                             hcalDigis = hcalDigis,
                                             muonDTDigis = muonDTDigis,
                                             #muonRPCDigis = muonRPCDigis,
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
    TrackerMergeType = cms.string('Digis'),  # kludge for fast simulation flag...
    # Merge Pileup Info?
    MergePileupInfo = cms.bool(True),                         
    # Use digis?               
    EcalMergeType = cms.string('Digis'),  # set to "Digis" to merge digis
    HcalMergeType = cms.string('Digis'),
    HcalDigiMerge = cms.string('FullProd'), #use sim hits for signal
    addMCDigiNoise = cms.untracked.bool(True),
    #
    # Input Specifications:
    #
    #
    # Tracking particles for validation
    #
    TrackingParticleLabelSig = cms.InputTag("mix","MergedTrackTruth"),
    StripDigiSimLinkLabelSig = cms.InputTag("simSiStripDigis"),
    PixelDigiSimLinkLabelSig = cms.InputTag("simSiPixelDigis"),
    DTDigiSimLinkLabelSig = cms.InputTag("simMuonDTDigis"),
    RPCDigiSimLinkLabelSig = cms.InputTag("simMuonRPCDigis","RPCDigiSimLink"),
    CSCStripDigiSimLinkLabelSig = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigiSimLinks"),
    CSCWireDigiSimLinkLabelSig = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigiSimLinks"),

    #                     
    PileupInfoInputTag = cms.InputTag("addPileupInfo"),
    BunchSpacingInputTag = cms.InputTag("addPileupInfo","bunchSpacing"),
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

                         
    # Validation
    TrackingParticlePileInputTag = cms.InputTag("mix","MergedTrackTruth"),
    StripDigiSimLinkPileInputTag = cms.InputTag("simSiStripDigis"),
    PixelDigiSimLinkPileInputTag = cms.InputTag("simSiPixelDigis"),
    DTDigiSimLinkPileInputTag = cms.InputTag("simMuonDTDigis"),
    RPCDigiSimLinkPileInputTag = cms.InputTag("simMuonRPCDigis","RPCDigiSimLink"),
    CSCStripDigiSimLinkPileInputTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigiSimLinks"),
    CSCWireDigiSimLinkPileInputTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigiSimLinks"),

    # Dead APV Vector
    SistripAPVPileInputTag = cms.InputTag("mix","AffectedAPVList"),
    SistripAPVLabelSig = cms.InputTag("mix","AffectedAPVList"),

    # Note: elements with "@MIXING" in the input tag are generated by
    # running Raw2Digi in the input step on the Secondary input stream
    EBPileInputTag = cms.InputTag("ecalDigis","ebDigis","@MIXING"),
    EEPileInputTag = cms.InputTag("ecalDigis","eeDigis","@MIXING"),
    ESPileInputTag = cms.InputTag("ecalPreshowerDigis","","@MIXING"),
    #ESPileInputTag = cms.InputTag("esRawToDigi","","@MIXING"),
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
    RPCPileInputTag       = cms.InputTag("simMuonRPCDigis",""),
#    RPCPileInputTag       = cms.InputTag("muonRPCDigis","","@MIXING"),  # use MC digis...
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
    TrackingParticleCollectionDM = cms.string('MergedTrackTruth'),
    StripDigiSimLinkCollectionDM = cms.string('StripDigiSimLink'),
    PixelDigiSimLinkCollectionDM = cms.string('PixelDigiSimLink'),
    DTDigiSimLinkDM = cms.string('simMuonDTDigis'),
    RPCDigiSimLinkDM = cms.string('RPCDigiSimLink'),
    CSCStripDigiSimLinkDM = cms.string('MuonCSCStripDigiSimLinks'),
    CSCWireDigiSimLinkDM = cms.string('MuonCSCWireDigiSimLinks'),
    SiStripAPVListDM = cms.string('SiStripAPVList'),

    #
    #  Calorimeter Digis
    #               
    EBDigiCollectionDM   = cms.string(''),
    EEDigiCollectionDM   = cms.string(''),
    ESDigiCollectionDM   = cms.string(''),
    HBHEDigiCollectionDM = cms.string(''),
    HODigiCollectionDM   = cms.string(''),
    HFDigiCollectionDM   = cms.string(''),
    ZDCDigiCollectionDM  = cms.string('')
)

mixData.doEB = cms.bool(True)
mixData.doEE = cms.bool(True)
mixData.doES = cms.bool(True)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    # from signal: mix tracks not strip or pixel digis
    mixData.TrackerMergeType = "tracks"
    import FastSimulation.Tracking.recoTrackAccumulator_cfi
    mixData.tracker = FastSimulation.Tracking.recoTrackAccumulator_cfi.recoTrackAccumulator.clone()
    mixData.tracker.pileUpTracks = cms.InputTag("mix","generalTracks")
    mixData.hitsProducer = "famosSimHits"

