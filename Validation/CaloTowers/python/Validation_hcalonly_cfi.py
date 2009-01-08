import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
from Configuration.StandardSequences.VtxSmearedGauss_cff import *
from SimG4Core.Application.g4SimHits_cfi import *
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import *
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigis_cfi import *
from Configuration.StandardSequences.MagneticField_cff import *
from Configuration.StandardSequences.MixingNoPileUp_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
GlobalTag.globaltag = 'IDEAL_30X::All'
from Configuration.StandardSequences.GeometryHCAL_cff import *

from DQMServices.Core.DQM_cfg import *
maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
MessageLogger = cms.Service("MessageLogger")

source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/1_4_x/mc_pi+100_etaphi44.root')
)

hcalDigiAnalyzer = cms.EDFilter("HcalDigiTester",
                                digiLabel = cms.InputTag("simHcalDigis"),
                                subpedvalue = cms.untracked.bool(True),
                                outputFile = cms.untracked.string('HcalDigisValidationHF.root'),
                                hcalselector = cms.untracked.string('HF')
                                )

hcalRecoAnalyzer = cms.EDFilter("HcalRecHitsValidation",
                                outputFile = cms.untracked.string('HcalRecHitsValidationHB.root'),
                                eventype = cms.untracked.string('single'),
                                mc = cms.untracked.string('yes'),
                                sign = cms.untracked.string('*'),
                                hcalselector = cms.untracked.string('HF'),
                                ecalselector = cms.untracked.string('no')
                                )

hcalTowerAnalyzer = cms.EDFilter("CaloTowersValidation",
                                 outputFile = cms.untracked.string('CaloTowersValidationHB.root'),
                                 CaloTowerCollectionLabel = cms.untracked.string('towerMaker'),
                                 hcalselector = cms.untracked.string('HB')
                                 )

DQM.collectorHost = ''


VtxSmeared.SigmaX = 0.00001
VtxSmeared.SigmaY = 0.00001
VtxSmeared.SigmaZ = 0.00001

g4SimHits.UseMagneticField = False

hbhereco.digiLabel = 'simHcalDigis'
horeco.digiLabel = 'simHcalDigis'
hfreco.digiLabel = 'simHcalDigis'
