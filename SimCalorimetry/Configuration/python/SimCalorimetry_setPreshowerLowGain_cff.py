import FWCore.ParameterSet.Config as cms

def customise(process):

    process.simEcalPreshowerDigis.ESNoiseSigma = cms.untracked.double(3.0)
    process.simEcalPreshowerDigis.ESGain = cms.untracked.int32(1)
    process.simEcalPreshowerDigis.ESMIPADC = cms.untracked.double(9.0)

    process.simEcalUnsuppressedDigis.ESGain = cms.int32(1)
    process.simEcalUnsuppressedDigis.ESNoiseSigma = cms.double(3.0)
    process.simEcalUnsuppressedDigis.ESMIPADC = cms.double(9.0) 

    return(process)

