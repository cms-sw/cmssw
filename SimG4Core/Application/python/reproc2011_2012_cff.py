import FWCore.ParameterSet.Config as cms

def customiseGeant4(process):

    process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/QGSP_FTFP_BERT_EML_New')

    # extended geometric acceptance (full CASTOR acceptance)

    process.g4SimHits.Generator.MinEtaCut = cms.double(-6.7)
    process.g4SimHits.Generator.MaxEtaCut = cms.double(6.7)

    # use HF shower library instead of GFlash parameterization

    process.g4SimHits.HCalSD.UseShowerLibrary = cms.bool(True)
    process.g4SimHits.HCalSD.UseParametrize = cms.bool(False)
    process.g4SimHits.HCalSD.UsePMTHits = cms.bool(False)
    process.g4SimHits.HCalSD.UseFibreBundleHits = cms.bool(False)

    return(process)
