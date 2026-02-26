import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.CaloTPG.tpScales_cff import tpScales

LSParameter =cms.untracked.PSet(
HcalFeatureHFEMBit= cms.bool(False),
Min_Long_Energy= cms.double(10),#makes a cut based on energy deposited in short vrs long
    Min_Short_Energy= cms.double(10),
    Long_vrs_Short_Slope= cms.double(100.2),
    Long_Short_Offset= cms.double(10.1))


simHcalTriggerPrimitiveDigis = cms.EDProducer("HcalTrigPrimDigiProducer",
    peakFilter = cms.bool(True),
    weights = cms.vdouble(1.0, 1.0), ##hardware algo        
    weightsQIE11 = cms.PSet(
        ieta1 = cms.vint32(255, 255),
        ieta2 = cms.vint32(255, 255),
        ieta3 = cms.vint32(255, 255),
        ieta4 = cms.vint32(255, 255),
        ieta5 = cms.vint32(255, 255),
        ieta6 = cms.vint32(255, 255),
        ieta7 = cms.vint32(255, 255),
        ieta8 = cms.vint32(255, 255),
        ieta9 = cms.vint32(255, 255),
        ieta10 = cms.vint32(255, 255),
        ieta11 = cms.vint32(255, 255),
        ieta12 = cms.vint32(255, 255),
        ieta13 = cms.vint32(255, 255),
        ieta14 = cms.vint32(255, 255),
        ieta15 = cms.vint32(255, 255),
        ieta16 = cms.vint32(255, 255),
        ieta17 = cms.vint32(255, 255),
        ieta18 = cms.vint32(255, 255),
        ieta19 = cms.vint32(255, 255),
        ieta20 = cms.vint32(255, 255),
        ieta21 = cms.vint32(255, 255),
        ieta22 = cms.vint32(255, 255),
        ieta23 = cms.vint32(255, 255),
        ieta24 = cms.vint32(255, 255),
        ieta25 = cms.vint32(255, 255),
        ieta26 = cms.vint32(255, 255),
        ieta27 = cms.vint32(255, 255),
        ieta28 = cms.vint32(255, 255)
    ),

    latency = cms.int32(1),
    FG_threshold = cms.uint32(12), ## threshold for setting fine grain bit
    overrideFGHF = cms.bool(False), ## switch: False = read thresholds from TPParameters (default), True = override with FG_HF_thresholds
    FG_HF_thresholds = cms.vuint32(17, 255), ## thresholds for setting fine grain bit
    ZS_threshold = cms.uint32(1),  ## threshold for setting TP zero suppression

    # To be used when overriding the CondDB, default is with vetoing off ("coded" threshold = 0)
    # To run PFA1' + vetoing with no threshold, use 2048
    # All other values (1, 2047) are interpreted literally as the PFA1' veto threshold 
    codedVetoThresholds = cms.PSet(
        ieta1  = cms.int32(0),
        ieta2  = cms.int32(0),
        ieta3  = cms.int32(0),
        ieta4  = cms.int32(0),
        ieta5  = cms.int32(0),
        ieta6  = cms.int32(0),
        ieta7  = cms.int32(0),
        ieta8  = cms.int32(0),
        ieta9  = cms.int32(0),
        ieta10 = cms.int32(0),
        ieta11 = cms.int32(0),
        ieta12 = cms.int32(0),
        ieta13 = cms.int32(0),
        ieta14 = cms.int32(0),
        ieta15 = cms.int32(0),
        ieta16 = cms.int32(0),
        ieta17 = cms.int32(0),
        ieta18 = cms.int32(0),
        ieta19 = cms.int32(0),
        ieta20 = cms.int32(0),
        ieta21 = cms.int32(0),
        ieta22 = cms.int32(0),
        ieta23 = cms.int32(0),
        ieta24 = cms.int32(0),
        ieta25 = cms.int32(0),
        ieta26 = cms.int32(0),
        ieta27 = cms.int32(0),
        ieta28 = cms.int32(0)
    ),

    overrideHBLLP = cms.bool(False), ## switch: False = read thresholds from TPParameters (default), True = override with HB_LLP_thresholds                                                         
    ## defaults for energy requirement for bits 12-15 are high / low to avoid FG bit 0-4 being set when not intended                                                                                              
    HB_LLP_thresholds = cms.vuint32(0, 0, 999, 999),  ## default energy thresholds for setting HB LLP bit                                                                                           
                                                      ## depths 1,2 max energy, depths 3+ min energy, prompt min energy, delayed min energy                                            

    overrideDBvetoThresholdsHB = cms.bool(False),
    overrideDBvetoThresholdsHE = cms.bool(False),
    numberOfSamples = cms.int32(4),
    numberOfPresamples = cms.int32(2),
    numberOfSamplesHF = cms.int32(4),
    numberOfPresamplesHF = cms.int32(2),
    numberOfFilterPresamplesHBQIE11 = cms.int32(0),
    numberOfFilterPresamplesHEQIE11 = cms.int32(0),
    useTDCInMinBiasBits = cms.bool(False), # TDC information not used in MB fine grain bits
    MinSignalThreshold = cms.uint32(0), # For HF PMT veto
    PMTNoiseThreshold = cms.uint32(0),  # For HF PMT veto
    LSConfig=LSParameter,

    upgradeHF = cms.bool(False),
    upgradeHB = cms.bool(False),
    upgradeHE = cms.bool(False),

    applySaturationFix = cms.bool(False), # Apply the TP energy saturation fix for Peak Finder Algorithm only for Run3 

    # parameters = cms.untracked.PSet(
    #     FGVersionHBHE=cms.uint32(0),
    #     TDCMask=cms.uint64(0xFFFFFFFFFFFFFFFF),
    #     ADCThreshold=cms.uint32(0),
    #     FGThreshold=cms.uint32(12)
    # ),

    #vdouble weights = { -1, -1, 1, 1} //low lumi algo
    # Input digi label (_must_ be without zero-suppression!)
    inputLabel = cms.VInputTag(cms.InputTag('simHcalUnsuppressedDigis'),
                               cms.InputTag('simHcalUnsuppressedDigis')),
    inputUpgradeLabel = cms.VInputTag(
        cms.InputTag('simHcalUnsuppressedDigis:HBHEQIE11DigiCollection'),
        cms.InputTag('simHcalUnsuppressedDigis:HFQIE10DigiCollection')),
    InputTagFEDRaw = cms.InputTag("rawDataCollector"),
    overrideDBweightsAndFilterHB = cms.bool(False),
    overrideDBweightsAndFilterHE = cms.bool(False),
    RunZS = cms.bool(False),
    FrontEndFormatError = cms.bool(False), # Front End Format Error, for real data only
    PeakFinderAlgorithm = cms.int32(2),

    tpScales = tpScales,
)

from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
run2_HE_2017.toModify(simHcalTriggerPrimitiveDigis, upgradeHE=True)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toModify(simHcalTriggerPrimitiveDigis,
                      upgradeHF=True,
                      numberOfSamplesHF = 2,
                      numberOfPresamplesHF = 1
)

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(simHcalTriggerPrimitiveDigis, upgradeHB=True)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(simHcalTriggerPrimitiveDigis, applySaturationFix=True)
