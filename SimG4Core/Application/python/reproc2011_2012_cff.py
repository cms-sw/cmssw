import FWCore.ParameterSet.Config as cms

def customiseG4(process):

    process.load('SimG4Core.Application.g4SimHits_cfi')

    process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/QGSP_FTFP_BERT_EML_New')

    # use HF shower library instead of GFlash parameterization

    process.g4SimHits.HCalSD.UseShowerLibrary = cms.bool(True)
    process.g4SimHits.HCalSD.UseParametrize = cms.bool(False)
    process.g4SimHits.HCalSD.UsePMTHits = cms.bool(False)
    process.g4SimHits.HCalSD.UseFibreBundleHits = cms.bool(False)

    return(process)
