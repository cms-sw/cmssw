import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *

# require generated tau to decay hadronically
objectTypeSelectedTauValDenominatorZTT = cms.EDFilter("TauGenJetDecayModeSelector",
     src = cms.InputTag("tauGenJets"),
     select = cms.vstring('oneProng0Pi0', 'oneProng1Pi0', 'oneProng2Pi0', 'oneProngOther',
                          'threeProng0Pi0', 'threeProng1Pi0', 'threeProngOther', 'rare'),
     filter = cms.bool(False)
)

#clones the kinematic selection
kinematicSelectedTauValDenominatorZTT = kinematicSelectedTauValDenominator.clone( src = cms.InputTag("objectTypeSelectedTauValDenominatorZTT") )

#labels TauValNumeratorAndDenominator modules in the sequence to match kinematicSelectedTauValDenominatorZTT and adds ZTT to the extention name
zttLabeler = lambda module : SetValidationAttributes(module, 'ZTT', 'kinematicSelectedTauValDenominatorZTT')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominator.visit(zttModifier)

#clones the whole sequence and related modules adding a ZTT to the end of the name, can even set the correct dependencies, but not needed here
import PhysicsTools.PatAlgos.tools.helpers as configtools
procAttributes = dir(proc)
configtools.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'ZTT')
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('ZTT') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

#clones the TauEfficiencies module
TauEfficienciesZTT = TauEfficiencies.clone( plots = Utils.SetPlotSequence(TauValNumeratorAndDenominatorZTT) )

#Define some useful sequences
produceDenominatorZTT = cms.Sequence(
      tauGenJets
      +objectTypeSelectedTauValDenominatorZTT
      +kinematicSelectedTauValDenominatorZTT
      )

runTauValidationBatchModeZTT = cms.Sequence(
      TauValNumeratorAndDenominatorZTT
      )

runTauValidationZTT = cms.Sequence(
      runTauValidationBatchModeZTT
      +TauEfficienciesZTT
      )

#Needed by RunValidation_cfg
validationBatch = cms.Path(produceDenominatorZTT*runTauValidationBatchModeZTT)
validationStd   = cms.Path(produceDenominatorZTT*runTauValidationZTT)
