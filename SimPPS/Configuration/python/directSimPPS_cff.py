import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from SimPPS.DirectSimProducer.ppsDirectProtonSimulation_cff import *

directSimPPSTask = cms.Task(
    beamDivergenceVtxGenerator,
    ppsDirectProtonSimulation
)

directSimPPS = cms.Sequence(directSimPPSTask)

# modify according to era

def _modify2016(process):
    print('Process customised for 2016 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2016_cfi')

def _modify2017(process):
    print('Process customised for 2017 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2017_cfi')

def _modify2018(process):
    print('Process customised for 2018 PPS era')
    process.load('SimPPS.DirectSimProducer.simPPS2018_cfi')

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
