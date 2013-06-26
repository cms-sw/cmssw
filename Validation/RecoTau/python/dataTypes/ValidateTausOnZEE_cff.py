import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

selectElectrons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = 11",
    "keep++ pdgId = -11",
    )
)

selectStableElectrons = genParticlesForJets.clone(src = cms.InputTag("selectElectrons"))

#objectTypeSelectedTauValDenominatorModule = copy.deepcopy(iterativeCone5GenJets)
#objectTypeSelectedTauValDenominatorModule.src = cms.InputTag("selectElectronsForGenJets")

kinematicSelectedTauValDenominatorZEE = cms.EDFilter(
   "TauValGenPRefSelector", #"GenJetSelector"
   src = cms.InputTag('selectStableElectrons'),
   cut = kinematicSelectedTauValDenominatorCut,#cms.string('pt > 5. && abs(eta) < 2.5'), #Defined: Validation.RecoTau.RecoTauValidation_cfi 
   filter = cms.bool(False)
)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'ZEE') #clones the sequence inside the process with ZEE postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'ZEE') #clones the sequence inside the process with ZEE postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorZEE, 'kinematicSelectedTauValDenominator', 'kinematicSelectedTauValDenominatorZEE') #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence ZEE to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'ZEE')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorZEE.visit(zttModifier)

#Sets the correct naming to efficiency histograms
proc.efficienciesZEE.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorZEE)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('ZEE') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorZEE = cms.Sequence(
    selectElectrons*
    selectStableElectrons*
    kinematicSelectedTauValDenominatorZEE
    )

produceDenominator = cms.Sequence(produceDenominatorZEE)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorZEE*
      TauValNumeratorAndDenominatorZEE
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode*
      TauEfficienciesZEE
      )

