import FWCore.ParameterSet.Config as cms

def customise_hcalNZS(process):
    if hasattr(process,'simHcalDigis'):
        process.simHcalDigis.useConfigZSvalues=cms.int32(1)
        process.simHcalDigis.HBlevel=cms.int32(-999)
        process.simHcalDigis.HElevel=cms.int32(-999)
        process.simHcalDigis.HOlevel=cms.int32(-999)
        process.simHcalDigis.HFlevel=cms.int32(-999)

    return process
