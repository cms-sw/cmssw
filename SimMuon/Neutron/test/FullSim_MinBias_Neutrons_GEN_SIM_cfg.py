#
# An example configuration for production of
# a SIM level sample of neutron hits in muon systems
#
# $Id: FullSim_MinBias_Neutrons_GEN_SIM_cfg.py,v 1.1 2010/08/20 00:32:43 khotilov Exp $

import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

################################################################################
# Basic running parameters (modify to your needs)

# name of the output file
theFileName = 'out_n.root'

# GEANT physics list:
physicsList = 'SimG4Core/Physics/QGSP_BERT_HP'
#physicsList = 'SimG4Core/Physics/QGSP_BERT_HP_EML'
#physicsList = 'SimG4Core/Physics/QGSP_BERT_EMLSN'
#physicsList = 'SimG4Core/Physics/QGSP_BERT_EML'

# run number
theRun = 30000


# CRAB overwrites the following:
# number of events to generate
theNumberOfEvents = 5

################################################################################
# some runtime config

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(theNumberOfEvents) )

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

#process.Timing = cms.Service("Timing")
#process.Tracer = cms.Service("Tracer")

################################################################################
# standard configuration imports

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.EventContent.EventContent_cff')


################################################################################
# Special neutrons-related imports

process.load('SimMuon.Neutron.GeneratorNeutrons_cff')

process.load('SimMuon.Neutron.SimIdealNeutrons_cff')
#process.load('SimMuon.Neutron.SimIdealNeutrons_NoQuads_cff')

process.g4SimHitsNeutrons.Physics.type = physicsList

process.load('SimMuon.Neutron.MinBias_7TeV_Neutrons_cfi')
#process.load('SimMuon.Neutron.MinBias_14TeV_Neutrons_cfi')

process.load('SimMuon.Neutron.VtxSmearedRealistic7TeVCollision_Neutrons_cff')
#process.load('SimMuon.Neutron.VtxSmearedGauss_Neutrons_cff')

# this one should be loaded last from the neutron-related imports:
process.load('SimMuon.Neutron.neutronSimHitsProcessing_cff')


################################################################################
# Global conditions tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#Global_Tags_for_Monte_Carlo_Prod

#process.GlobalTag.globaltag = 'MC_36Y_V10::All'
process.GlobalTag.globaltag = 'MC_38Y_V9::All'


################################################################################
# seeding the random engine

from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
#randSvc.resetSeeds(theRun)
randSvc.populate()

#print "random service:", process.RandomNumberGeneratorService
#print "source:", process.source


################################################################################
# Source definition

process.source = cms.Source("EmptySource",firstRun = cms.untracked.uint32(theRun))


################################################################################
# Output definition
# - keep the original g4SimHitsNeutrons muon simhits 
#   which have "g4SimHitsNeutrons" module labels
# - keep all the output of NeutronHitsCollector and EmptyHepMCProducer
#   which has module labels "g4SimHits" and generator respectively
# the latter should be enough to make MixingModule happy with this input

#process.load("Configuration.EventContent.EventContent_cff")
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = cms.untracked.vstring('drop *',
        #'keep PSimHits_*NeutronWriter_*_*',
        'keep *_g4SimHits_*_*',
        'keep *_generator_*_*',
        'keep *_*_Muon*Hits_*'),
    fileName = cms.untracked.string(theFileName),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('simulation_step')
    )
)


################################################################################
# Path and EndPath definitions

process.generation_step = cms.Path(process.pgen_neutrons)
process.simulation_step = cms.Path(process.psim_neutrons)
process.neutron_simhits_step = cms.Path(process.neutron_simhits_seq)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)


################################################################################
# Schedule definition

process.schedule = cms.Schedule(
    process.generation_step,
    process.simulation_step,
    process.neutron_simhits_step,
    process.endjob_step,
    process.out_step
)

# special treatment in case of production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generatorNeutrons*getattr(process,path)._seq
