import FWCore.ParameterSet.Config as cms

def customise(process):
    process.load('sample')
    import Validation.RecoMuon.RelValCustoms as switch
    #switch.harvest_only(process)
    #switch_validation_only(process)
    return(process)
