import FWCore.ParameterSet.Config as cms

ecal_digi_parameters = cms.PSet(
    EEdigiCollection = cms.string(''),
    EBdigiCollection = cms.string(''),
    ESdigiCollection = cms.string(''),
    UseLCcorrection  = cms.untracked.bool(True),

    EBCorrNoiseMatrixG12 = cms.vdouble (
    1.00000, 0.71073, 0.55721, 0.46089, 0.40449,
    0.35931, 0.33924, 0.32439, 0.31581, 0.30481 ) ,

    EECorrNoiseMatrixG12 = cms.vdouble (
    1.00000, 0.71373, 0.44825, 0.30152, 0.21609,
    0.14786, 0.11772, 0.10165, 0.09465, 0.08098 ) ,

    EBCorrNoiseMatrixG06 = cms.vdouble (
    1.00000, 0.70946, 0.58021, 0.49846, 0.45006,
    0.41366, 0.39699, 0.38478, 0.37847, 0.37055 ) ,

    EECorrNoiseMatrixG06 = cms.vdouble (
    1.00000, 0.71217, 0.47464, 0.34056, 0.26282,
    0.20287, 0.17734, 0.16256, 0.15618, 0.14443 ),

    EBCorrNoiseMatrixG01 = cms.vdouble (
    1.00000, 0.73354, 0.64442, 0.58851, 0.55425,
    0.53082, 0.51916, 0.51097, 0.50732, 0.50409 ) ,

    EECorrNoiseMatrixG01 = cms.vdouble (
    1.00000, 0.72698, 0.62048, 0.55691, 0.51848,
    0.49147, 0.47813, 0.47007, 0.46621, 0.46265 ) ,

    EcalPreMixStage1 = cms.bool(False),
    EcalPreMixStage2 = cms.bool(False)
    
)

