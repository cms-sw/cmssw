import FWCore.ParameterSet.Config as cms

def customise_jets(process):
    if hasattr(process, 'reconstruction_step'):
        process=customise_RECO(process)
    return process
    
def customise_RECO(process):
    if hasattr(process, 'recoJetIds'):
        process.reconstruction_step.remove(process.recoJetIds)
    return process
        
