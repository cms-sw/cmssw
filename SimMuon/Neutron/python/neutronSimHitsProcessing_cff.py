import FWCore.ParameterSet.Config as cms

################################################################################
# sub-system NeutronWriters
from SimMuon.CSCDigitizer.cscNeutronWriter_cfi import *
from SimMuon.RPCDigitizer.rpcNeutronWriter_cfi import *
from SimMuon.DTDigitizer.dtNeutronWriter_cfi import *

cscNeutronWriter.input = cms.InputTag("g4SimHitsNeutrons","MuonCSCHits")
rpcNeutronWriter.input = cms.InputTag("g4SimHitsNeutrons","MuonRPCHits")
dtNeutronWriter.input  = cms.InputTag("g4SimHitsNeutrons","MuonDTHits")

################################################################################
# Special utility modules for neutron collections processing
# and making MixingModule happy.
# They define new "generator" and new "g4SimHits" modules

from SimMuon.Neutron.emptyHepMCProducer_cfi import *
from SimMuon.Neutron.neutronHitsCollector_cfi import *


################################################################################
# extending the random number engine

from Configuration.StandardSequences.Services_cff import RandomNumberGeneratorService
RandomNumberGeneratorService.generatorNeutrons = cms.PSet( initialSeed = cms.untracked.uint32(1234), engineName = cms.untracked.string('TRandom3') )
RandomNumberGeneratorService.g4SimHitsNeutrons = cms.PSet( initialSeed = cms.untracked.uint32(1234), engineName = cms.untracked.string('TRandom3') )
RandomNumberGeneratorService.cscNeutronWriter  = cms.PSet( initialSeed = cms.untracked.uint32(1234), engineName = cms.untracked.string('TRandom3') )
RandomNumberGeneratorService.rpcNeutronWriter  = cms.PSet( initialSeed = cms.untracked.uint32(1234), engineName = cms.untracked.string('TRandom3') )
RandomNumberGeneratorService.dtNeutronWriter   = cms.PSet( initialSeed = cms.untracked.uint32(1234), engineName = cms.untracked.string('TRandom3') )


################################################################################
# processing sequence

neutron_simhits_seq = cms.Sequence((cscNeutronWriter + rpcNeutronWriter + dtNeutronWriter) * (generator + g4SimHits))

