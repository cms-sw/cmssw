import FWCore.ParameterSet.Config as cms

def customiseEcalSD(process):

    # fragment allowing to simulate correct Ecal hit depth 
    process.g4SimHits.ECalSD.IgnoreDepthCorr = cms.bool(False)

    return(process)
