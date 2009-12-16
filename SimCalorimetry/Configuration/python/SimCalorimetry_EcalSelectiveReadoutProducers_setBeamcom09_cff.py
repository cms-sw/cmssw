import FWCore.ParameterSet.Config as cms

def customise(process):

    # ECAL TPG with 2009 beam commissioning TTF thresholds

    process.EcalTrigPrimESProducer.DatabaseFile = cms.untracked.string('TPG_1x1_1GeV_3x3_2GeV.txt.gz')
    
    # ECAL SRP settings for 2009 beam commissioning

    process.simEcalDigis.ecalDccZs1stSample = cms.int32(3)
    process.simEcalDigis.dccNormalizedWeights = cms.vdouble(-1.1865, 0.0195, 0.2900, 0.3477, 0.3008, 0.2266)
    process.simEcalDigis.srpBarrelLowInterestChannelZS = cms.double(2.25*.035)
    process.simEcalDigis.srpEndcapLowInterestChannelZS = cms.double(3.75*0.06)

    return(process)

