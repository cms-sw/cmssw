import FWCore.ParameterSet.Config as cms

def customise(process):

    # advanced Geant4 physics
    #process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/QGSP_FTFP_BERT_EML_New')

    # extended geometric acceptance (full CASTOR acceptance)
    #process.g4SimHits.Generator.MinEtaCut = cms.double(-6.7)
    #process.g4SimHits.Generator.MaxEtaCut = cms.double(6.7)

    # Russian roulette parameters
    process.g4SimHits.Physics.RusRoGammaEnergyLimit = cms.double(5.)
    process.g4SimHits.Physics.RusRoEcalGamma     = cms.double(0.3)
    process.g4SimHits.Physics.RusRoHcalGamma     = cms.double(0.1)
    process.g4SimHits.Physics.RusRoMuonIronGamma = cms.double(0.1)
    process.g4SimHits.Physics.RusRoPreShowerGamma= cms.double(0.3)
    process.g4SimHits.Physics.RusRoWorldGamma    = cms.double(0.1)

    process.g4SimHits.StackingAction.RusRoEcalNeutronLimit = cms.double(10.)
    process.g4SimHits.StackingAction.RusRoEcalNeutron     = cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoHcalNeutron     = cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoMuonIronNeutron = cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoPreShowerNeutron= cms.double(0.1)
    process.g4SimHits.StackingAction.RusRoWorldNeutron    = cms.double(0.1)

    return(process)
