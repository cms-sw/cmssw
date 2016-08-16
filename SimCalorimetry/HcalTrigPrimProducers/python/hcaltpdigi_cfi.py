import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import hfTPScaleShift

LSParameter =cms.untracked.PSet(
HcalFeatureHFEMBit= cms.bool(False),
Min_Long_Energy= cms.double(10),#makes a cut based on energy deposited in short vrs long
    Min_Short_Energy= cms.double(10),
    Long_vrs_Short_Slope= cms.double(100.2),
    Long_Short_Offset= cms.double(10.1))


simHcalTriggerPrimitiveDigis = cms.EDProducer("HcalTrigPrimDigiProducer",
    peakFilter = cms.bool(True),
    weights = cms.vdouble(1.0, 1.0), ##hardware algo        
    latency = cms.int32(1),
    FG_MinimumBias = cms.bool(True),
    FG_threshold = cms.uint32(12), ## threshold for setting fine grain bit
    ZS_threshold = cms.uint32(1),  ## threshold for setting fine grain bit
    numberOfSamples = cms.int32(4),
    numberOfPresamples = cms.int32(2),
    numberOfSamplesHF = cms.int32(4),
    numberOfPresamplesHF = cms.int32(2),
    MinSignalThreshold = cms.uint32(0), # For HF PMT veto
    PMTNoiseThreshold = cms.uint32(0),  # For HF PMT veto
    LSConfig=LSParameter,
    
    
#
    #vdouble weights = { -1, -1, 1, 1} //low lumi algo
    # Input digi label (_must_ be without zero-suppression!)
    inputLabel = cms.VInputTag(cms.InputTag('simHcalUnsuppressedDigis'),
                               cms.InputTag('simHcalUnsuppressedDigis')),
    InputTagFEDRaw = cms.InputTag("rawDataCollector"),
    RunZS = cms.bool(False),
    FrontEndFormatError = cms.bool(False), # Front End Format Error, for real data only
    PeakFinderAlgorithm = cms.int32(2),

    HFTPScaleShift = hfTPScaleShift,
)
