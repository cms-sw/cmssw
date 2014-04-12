import FWCore.ParameterSet.Config as cms

ElectronMaterialEffects = cms.ESProducer("GsfMaterialEffectsESProducer",
    BetheHeitlerParametrization = cms.string('BetheHeitler_cdfmom_nC6_O5.par'),
    EnergyLossUpdator = cms.string('GsfBetheHeitlerUpdator'),
    ComponentName = cms.string('ElectronMaterialEffects'),
    MultipleScatteringUpdator = cms.string('MultipleScatteringUpdator'),
    Mass = cms.double(0.000511),
    BetheHeitlerCorrection = cms.int32(2)
)


