import FWCore.ParameterSet.Config as cms

def customise(process):

    process.simEcalPreshowerDigis.ESNoiseSigma = cms.untracked.double(6)
    process.simEcalPreshowerDigis.ESGain = cms.untracked.int32(2)
    process.simEcalPreshowerDigis.ESMIPADC = cms.untracked.double(55)

    process.mix.digitizers.ecal.ESGain = cms.int32(2)
    process.mix.digitizers.ecal.ESNoiseSigma = cms.double(6)
    process.mix.digitizers.ecal.ESMIPADC = cms.double(55) 
#    process.simEcalUnsuppressedDigis.ESGain = cms.int32(2)
#    process.simEcalUnsuppressedDigis.ESNoiseSigma = cms.double(6)
#    process.simEcalUnsuppressedDigis.ESMIPADC = cms.double(55) 

    return(process)

