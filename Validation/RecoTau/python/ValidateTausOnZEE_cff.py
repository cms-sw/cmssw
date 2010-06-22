import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
selectElectrons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = 11",
    "keep++ pdgId = -11",
    )
)

selectElectronsForGenJets = copy.deepcopy(genParticlesForJets)
selectElectronsForGenJets.src = cms.InputTag("selectElectrons")

objectTypeSelectedTauValDenominator = copy.deepcopy(iterativeCone5GenJets)
objectTypeSelectedTauValDenominator.src = cms.InputTag("selectElectronsForGenJets")

produceDenominator = cms.Sequence(
      selectElectrons
      +selectElectronsForGenJets
      +objectTypeSelectedTauValDenominator
      +kinematicSelectedTauValDenominator
      )

runTauValidationBatchMode = cms.Sequence(
      produceDenominator
      +TauValNumeratorAndDenominator
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficiencies
      )

