import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

selectMuons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = 13",
    "keep++ pdgId = -13",
    )
)

selectStableMuons = genParticlesForJets.clone(src = cms.InputTag("selectMuons"))

kinematicSelectedTauValDenominatorZMM = cms.EDFilter(
   "TauValGenPRefSelector", #"GenJetSelector"
   src = cms.InputTag('selectStableMuons'),
   cut = kinematicSelectedTauValDenominatorCut,#cms.string('pt > 5. && abs(eta) < 2.5'), #Defined: Validation.RecoTau.RecoTauValidation_cfi 
   filter = cms.bool(False)
)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'ZMM') #clones the sequence inside the process with ZMM postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'ZMM') #clones the sequence inside the process with ZMM postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorZMM, 'kinematicSelectedTauValDenominator', 'kinematicSelectedTauValDenominatorZMM') #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence ZMM to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'ZMM')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorZMM.visit(zttModifier)

#Sets the correct naming to efficiency histograms
proc.efficienciesZMM.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorZMM)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('ZMM') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorZMM = cms.Sequence(
      selectMuons
      +selectStableMuons
#      +objectTypeSelectedTauValDenominatorModule
      +kinematicSelectedTauValDenominatorZMM
      )

produceDenominator = cms.Sequence(produceDenominatorZMM)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorZMM
      +TauValNumeratorAndDenominatorZMM
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesZMM
      )

