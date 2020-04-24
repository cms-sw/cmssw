import FWCore.ParameterSet.Config as cms

def customise_hcalNZS(process):
    if hasattr(process,'simHcalDigis'):
        process.simHcalDigis.markAndPass = cms.bool(True)

    return process
