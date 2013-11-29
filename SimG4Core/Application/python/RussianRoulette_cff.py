import FWCore.ParameterSet.Config as cms

def customise(process):

    # advanced Geant4 physics
    #process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/QGSP_FTFP_BERT_EML_New')

    # extended geometric acceptance (full CASTOR acceptance)
    #process.g4SimHits.Generator.MinEtaCut = cms.double(-6.7)
    #process.g4SimHits.Generator.MaxEtaCut = cms.double(6.7)

    # Russian roulette enabled
    process.g4SimHits.StackingAction.RusRoGammaEnergyLimit = cms.double(5.0)
    process.g4SimHits.StackingAction.RusRoNeutronEnergyLimit = cms.double(10.0)

    return(process)
