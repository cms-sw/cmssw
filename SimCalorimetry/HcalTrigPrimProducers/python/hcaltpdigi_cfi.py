import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import tpScales
from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB

LSParameter =cms.untracked.PSet(
HcalFeatureHFEMBit= cms.bool(False),
Min_Long_Energy= cms.double(10),#makes a cut based on energy deposited in short vrs long
    Min_Short_Energy= cms.double(10),
    Long_vrs_Short_Slope= cms.double(100.2),
    Long_Short_Offset= cms.double(10.1))


simHcalTriggerPrimitiveDigis = cms.EDProducer("HcalTrigPrimDigiProducer",
    peakFilter = cms.bool(True),
    weights = cms.vdouble(1.0, 1.0), ##hardware algo        
    weightsQIE11 = cms.untracked.PSet(**dict([
        ("1",  cms.untracked.vdouble(1.0, 1.0)),
        ("2",  cms.untracked.vdouble(1.0, 1.0)),
        ("3",  cms.untracked.vdouble(1.0, 1.0)),
        ("4",  cms.untracked.vdouble(1.0, 1.0)),
        ("5",  cms.untracked.vdouble(1.0, 1.0)),
        ("6",  cms.untracked.vdouble(1.0, 1.0)),
        ("7",  cms.untracked.vdouble(1.0, 1.0)),
        ("8",  cms.untracked.vdouble(1.0, 1.0)),
        ("9",  cms.untracked.vdouble(1.0, 1.0)),
        ("10", cms.untracked.vdouble(1.0, 1.0)),
        ("11", cms.untracked.vdouble(1.0, 1.0)),
        ("12", cms.untracked.vdouble(1.0, 1.0)),
        ("13", cms.untracked.vdouble(1.0, 1.0)),
        ("14", cms.untracked.vdouble(1.0, 1.0)),
        ("15", cms.untracked.vdouble(1.0, 1.0)),
        ("16", cms.untracked.vdouble(1.0, 1.0)),
        ("17", cms.untracked.vdouble(1.0, 1.0)),
        ("18", cms.untracked.vdouble(1.0, 1.0)),
        ("19", cms.untracked.vdouble(1.0, 1.0)),
        ("20", cms.untracked.vdouble(1.0, 1.0)),
        ("21", cms.untracked.vdouble(1.0, 1.0)),
        ("22", cms.untracked.vdouble(1.0, 1.0)),
        ("23", cms.untracked.vdouble(1.0, 1.0)),
        ("24", cms.untracked.vdouble(1.0, 1.0)),
        ("25", cms.untracked.vdouble(1.0, 1.0)),
        ("26", cms.untracked.vdouble(1.0, 1.0)),
        ("27", cms.untracked.vdouble(1.0, 1.0)),
        ("28", cms.untracked.vdouble(1.0, 1.0))
        ])
    ),

    latency = cms.int32(1),
    FG_threshold = cms.uint32(12), ## threshold for setting fine grain bit
    FG_HF_thresholds = cms.vuint32(17, 255), ## thresholds for setting fine grain bit
    ZS_threshold = cms.uint32(1),  ## threshold for setting TP zero suppression
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
    RunZS = cms.bool(False),
    FrontEndFormatError = cms.bool(False), # Front End Format Error, for real data only
    PeakFinderAlgorithm = cms.int32(2),

    tpScales = tpScales,
)

run2_HE_2017.toModify(simHcalTriggerPrimitiveDigis, upgradeHE=cms.bool(True))
run2_HF_2017.toModify(simHcalTriggerPrimitiveDigis, 
                      upgradeHF=cms.bool(True),
                      numberOfSamplesHF = cms.int32(2),
                      numberOfPresamplesHF = cms.int32(1)
)
run2_HF_2017.toModify(tpScales.HF, NCTShift=cms.int32(2))
run3_HB.toModify(simHcalTriggerPrimitiveDigis, upgradeHB=cms.bool(True))
run3_HB.toModify(tpScales.HBHE, LSBQIE11Overlap=cms.double(1/16.))
