import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

objectTypeSelectedTauValDenominatorQCD = iterativeCone5GenJets.clone()

#clones the kinematic selection
kinematicSelectedTauValDenominatorQCD = kinematicSelectedTauValDenominator.clone( src = cms.InputTag("objectTypeSelectedTauValDenominatorQCD") )

#labels TauValNumeratorAndDenominator modules in the sequence to match kinematicSelectedTauValDenominatorQCD and adds QCD to the extention name
zttLabeler = lambda module : SetValidationAttributes(module, 'QCD', 'kinematicSelectedTauValDenominatorQCD')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominator.visit(zttModifier)

#clones the whole sequence and related modules adding a QCD to the end of the name, can even set the correct dependencies, but not needed here
import PhysicsTools.PatAlgos.tools.helpers as configtools
procAttributes = dir(proc)
configtools.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'QCD')
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('QCD') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

#clones the TauEfficiencies module
TauEfficienciesQCD = TauEfficiencies.clone( plots = Utils.SetPlotSequence(TauValNumeratorAndDenominatorQCD) )

#Define some useful sequences
produceDenominatorQCD = cms.Sequence(
      genParticlesForJets
      +objectTypeSelectedTauValDenominatorQCD
      +kinematicSelectedTauValDenominatorQCD
      )

runTauValidationBatchModeQCD = cms.Sequence(
      TauValNumeratorAndDenominatorQCD
      )

runTauValidationQCD = cms.Sequence(
      runTauValidationBatchModeQCD
      +TauEfficienciesQCD
      )

#Needed by RunValidation_cfg
validationBatch = cms.Path(produceDenominatorQCD*runTauValidationBatchModeQCD)
validationStd   = cms.Path(produceDenominatorQCD*runTauValidationQCD)

