import FWCore.ParameterSet.Config as cms

#--- reset HB/HE digi frame size to 8TS & SOI=3 (0,1,2,3)
#--- NB: needs ZS re-adjustment/customization at the same time 

def customise(process):
    process.mix.digitizers.hcal.hb.readoutFrameSize = 8
    process.mix.digitizers.hcal.he.readoutFrameSize = 8
    process.mix.digitizers.hcal.hb.binOfMaximum = cms.int32(4)
    process.mix.digitizers.hcal.he.binOfMaximum = cms.int32(4)
    return(process)
