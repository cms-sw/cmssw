import FWCore.ParameterSet.Config as cms

def customise(process):

    # fragment allowing to simulate neutron background in muon system

    # time window 1 millisecond
    process.common_maximum_time.MaxTrackTime = cms.double(1000000.0)

    # Physics List HP
    process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/FTFP_BERT_HP_EML')

    # Russian roulette disabled
    process.g4SimHits.StackingAction.RusRoGammaEnergyLimit = cms.double(0.0)
    process.g4SimHits.StackingAction.RusRoNeutronEnergyLimit = cms.double(0.0)

    return(process)
