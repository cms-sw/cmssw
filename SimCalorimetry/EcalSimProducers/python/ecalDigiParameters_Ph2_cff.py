import FWCore.ParameterSet.Config as cms

ecal_digi_parameters = cms.PSet(
    EBdigiCollectionPh2 = cms.string(''),
    UseLCcorrection  = cms.untracked.bool(True),

    # Phase2 noise correlation matrices for G10 from 2023 TB
    EBCorrNoiseMatrixG10Ph2 = cms.vdouble (
    1.0000, 0.6986, 0.3782, 0.2452, 0.2900, 0.3532, 0.3911, 0.3896,
    0.3892, 0.3842, 0.3796, 0.3711, 0.3763, 0.3791, 0.3867, 0.3820),

    # G1 numbers duplicated from G10 in lack of measured G1 correlations
    EBCorrNoiseMatrixG01Ph2 = cms.vdouble (
    1.0000, 0.6986, 0.3782, 0.2452, 0.2900, 0.3532, 0.3911, 0.3896,
    0.3892, 0.3842, 0.3796, 0.3711, 0.3763, 0.3791, 0.3867, 0.3820),

    EcalPreMixStage1 = cms.bool(False)
)

from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toModify(ecal_digi_parameters, EcalPreMixStage1 = True)
