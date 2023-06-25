import FWCore.ParameterSet.Config as cms

component_digi_parameters = cms.PSet(
    componentDigiTag = cms.string("Component"),
    componentTimeTag = cms.string("Component"),
    componentSeparateDigi = cms.bool(False),
    componentAddToBarrel  = cms.bool(False),
    componentTimePhase  = cms.double(0.)
)

from Configuration.Eras.Modifier_ecal_component_cff import ecal_component
from Configuration.Eras.Modifier_ecal_component_finely_sampled_waveforms_cff import ecal_component_finely_sampled_waveforms
(ecal_component | ecal_component_finely_sampled_waveforms).toModify(component_digi_parameters,componentSeparateDigi=True)
