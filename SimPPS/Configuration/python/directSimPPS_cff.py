import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from SimPPS.DirectSimProducer.ppsDirectProtonSimulation_cff import *

directSimPPSTask = cms.Task(
    beamDivergenceVtxGenerator,
    ppsDirectProtonSimulation
)

directSimPPS = cms.Sequence(directSimPPSTask)

def unshiftVertex(process, smearingParams):
    """Undo vertex smearing using the parameters used for the sample production"""
    if not hasattr(process, 'ctppsBeamParametersFromLHCInfoESSource'):
        return
    from importlib import import_module
    _params = import_module('IOMC.EventVertexGenerators.VtxSmearedParameters_cfi')
    if not hasattr(_params, smearingParams):
        raise ImportError('Failed to import {} from vertex smearing parameters!'.format(smearingParams))
    _params = getattr(_params, smearingParams)
    process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetX45 = cms.double(-_params.X0.value())
    process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetY45 = cms.double(-_params.Y0.value())
    process.ctppsBeamParametersFromLHCInfoESSource.vtxOffsetZ45 = cms.double(-_params.Z0.value())

# modify according to era

def _modify2016(process):
    print('Process customised for 2016 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2016_cfi')
    unshiftVertex(process, 'Realistic25ns13TeV2016CollisionVtxSmearingParameters')

def _modify2017(process):
    print('Process customised for 2017 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2017_cfi')
    unshiftVertex(process, 'Realistic25ns13TeVEarly2017CollisionVtxSmearingParameters')

def _modify2018(process):
    print('Process customised for 2018 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2018_cfi')
    unshiftVertex(process, 'Realistic25ns13TeVEarly2018CollisionVtxSmearingParameters')

def _modify2022(process):
    print('Process customised for 2022 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2022_cfi')
    if hasattr(process, 'generator'):
        process.generator.energy = process.profile_2022_default.ctppsLHCInfo.beamEnergy
    if hasattr(process, 'ctppsGeometryESModule'):
        # replaced by the composite ESSource
        delattr(process, 'ctppsGeometryESModule')

modifyConfigurationStandardSequencesFor2016_ = eras.ctpps_2016.makeProcessModifier(_modify2016)
modifyConfigurationStandardSequencesFor2017_ = eras.ctpps_2017.makeProcessModifier(_modify2017)
modifyConfigurationStandardSequencesFor2018_ = eras.ctpps_2018.makeProcessModifier(_modify2018)
modifyConfigurationStandardSequencesFor2022_ = eras.ctpps_2022.makeProcessModifier(_modify2022)
