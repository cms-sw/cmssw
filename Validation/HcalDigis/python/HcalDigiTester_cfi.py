import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Simulation_cff import *
from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
from Configuration.StandardSequences.Geometry_cff import *
from Configuration.StandardSequences.Digi_cff     import *
from Configuration.StandardSequences.MixingNoPileUp_cff import *
from Configuration.StandardSequences.FakeConditions_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *
from DQMServices.Core.DQM_cfg import *

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalDigis/data/1_8_0/simhits_pi+100_etaphi44.root')
)

MessageLogger = cms.Service("MessageLogger")

hcalDigiAnalyzer = cms.EDFilter("HcalDigiTester",
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis"),
    zside = cms.untracked.string('*'),
    outputFile = cms.untracked.string('HcalDigisValidationHB.root'),
    hcalselector = cms.untracked.string('HB')
)

DQM.collectorHost = ''

simEcalUnsuppressedDigis.doNoise = False
simEcalUnsuppressedDigis.doESNoise = False

p = cms.Path(mix*calDigi*hcalDigiAnalyzer)

