import FWCore.ParameterSet.Config as cms

ecal_digi_parameters = cms.PSet(
    EBdigiCollectionPh2 = cms.string(''),
    UseLCcorrection  = cms.untracked.bool(True),

    #NOTE: Phase2 noise correlation matrices with fake numbers to simply test the code flow.

    EBCorrNoiseMatrixG10Ph2 = cms.vdouble (
    1.00000, 0.71073, 0.55721, 0.46089, 0.40449,
    0.35931, 0.33924, 0.32439, 0.31581, 0.30481, 0.40449,0.40449,0.40449,0.40449,0.40449,0.40449) ,

    EBCorrNoiseMatrixG01Ph2 = cms.vdouble (
    1.00000, 0.73354, 0.64442, 0.58851, 0.55425,
    0.53082, 0.51916, 0.51097, 0.50732, 0.50409, 0.40449,0.40449,0.40449,0.40449,0.40449,0.40449) ,

    EcalPreMixStage1 = cms.bool(False),
    EcalPreMixStage2 = cms.bool(False)
    
)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(ecal_digi_parameters, EcalPreMixStage1 = True)
