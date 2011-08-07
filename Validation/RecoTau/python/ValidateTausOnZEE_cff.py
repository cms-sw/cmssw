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

selectElectronsForGenJets = genParticlesForJets.clone( src = cms.InputTag("selectElectrons"))

objectTypeSelectedTauValDenominatorZEE = iterativeCone5GenJets.clone(src = cms.InputTag("selectElectronsForGenJets"))

#clones the kinematic selection
kinematicSelectedTauValDenominatorZEE = kinematicSelectedTauValDenominator.clone( src = cms.InputTag("objectTypeSelectedTauValDenominatorZEE") )

#labels TauValNumeratorAndDenominator modules in the sequence to match kinematicSelectedTauValDenominatorZEE and adds ZEE to the extention name
zttLabeler = lambda module : SetValidationAttributes(module, 'ZEE', 'kinematicSelectedTauValDenominatorZEE')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominator.visit(zttModifier)

#clones the whole sequence and related modules adding a ZEE to the end of the name, can even set the correct dependencies, but not needed here
import PhysicsTools.PatAlgos.tools.helpers as configtools
procAttributes = dir(proc)
configtools.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'ZEE')
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('ZEE') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

#clones the TauEfficiencies module
TauEfficienciesZEE = TauEfficiencies.clone( plots = Utils.SetPlotSequence(TauValNumeratorAndDenominatorZEE) )

#Define some useful sequences
produceDenominatorZEE = cms.Sequence(
      selectElectrons
      +selectElectronsForGenJets
      +objectTypeSelectedTauValDenominatorZEE
      +kinematicSelectedTauValDenominatorZEE
      )

runTauValidationBatchModeZEE = cms.Sequence(
      produceDenominatorZEE
      +TauValNumeratorAndDenominatorZEE
      )

runTauValidationZEE = cms.Sequence(
      runTauValidationBatchModeZEE
      +TauEfficienciesZEE
      )

#Needed by RunValidation_cfg
validationBatch = cms.Path(produceDenominatorZEE*runTauValidationBatchModeZEE)
validationStd   = cms.Path(produceDenominatorZEE*runTauValidationZEE)

