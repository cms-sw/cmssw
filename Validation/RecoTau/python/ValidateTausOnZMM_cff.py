import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
selectMuons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = 13",
    "keep++ pdgId = -13",
    )
)


selectMuonsForGenJets = genParticlesForJets.clone(src = cms.InputTag("selectMuons"))

objectTypeSelectedTauValDenominatorZMM = iterativeCone5GenJets.clone(src = cms.InputTag("selectMuonsForGenJets"))


#clones the kinematic selection
kinematicSelectedTauValDenominatorZMM = kinematicSelectedTauValDenominator.clone( src = cms.InputTag("objectTypeSelectedTauValDenominatorZMM") )

#labels TauValNumeratorAndDenominator modules in the sequence to match kinematicSelectedTauValDenominatorZMM and adds ZMM to the extention name
zttLabeler = lambda module : SetValidationAttributes(module, 'ZMM', 'kinematicSelectedTauValDenominatorZMM')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominator.visit(zttModifier)

#clones the whole sequence and related modules adding a ZMM to the end of the name, can even set the correct dependencies, but not needed here
import PhysicsTools.PatAlgos.tools.helpers as configtools
procAttributes = dir(proc)
configtools.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'ZMM')
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('ZMM') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

#clones the TauEfficiencies module
TauEfficienciesZMM = TauEfficiencies.clone( plots = Utils.SetPlotSequence(TauValNumeratorAndDenominatorZMM) )

#Define some useful sequences
produceDenominatorZMM = cms.Sequence(
      selectMuons
      +selectMuonsForGenJets
      +objectTypeSelectedTauValDenominatorZMM
      +kinematicSelectedTauValDenominatorZMM
      )

runTauValidationBatchModeZMM = cms.Sequence(
      TauValNumeratorAndDenominatorZMM
      )

runTauValidationZMM = cms.Sequence(
      runTauValidationBatchModeZMM
      +TauEfficienciesZMM
      )

#Needed by RunValidation_cfg
validationBatch = cms.Path(produceDenominatorZMM*runTauValidationBatchModeZMM)
validationStd   = cms.Path(produceDenominatorZMM*runTauValidationZMM)
