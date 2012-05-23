import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from RecoJets.Configuration.RecoPFJets_cff import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

PrimaryVertexFilter = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
    filter = cms.bool(False)
    )

TauMETSelector = cms.EDFilter(
    "METSelector",
    src    = cms.InputTag("pfMet"),
    minEt  = cms.double(25),
    maxEt  = cms.double(70),
    filter = cms.bool(False)
    )

selectedMuons = cms.EDFilter(
    "MuonSelector",
    src = cms.InputTag('muons'),
    cut = cms.string("pt > 20.0 && abs(eta) < 2.1 && isGlobalMuon = 1 && isTrackerMuon = 1"),
    filter = cms.bool(False)
	)

selectedMuonsIso = cms.EDFilter(
    "MuonSelector",
    src = cms.InputTag('selectedMuons'),
    cut = cms.string('(isolationR03().emEt + isolationR03().hadEt + isolationR03().sumPt)/pt < 0.10'),
    filter = cms.bool(False)
	)    

selectedTaus = cms.EDFilter(
    "TauValTauSelector",
    src = cms.InputTag('hpsPFTauProducer'),
    cut = cms.string("pt > 20.0 && abs(eta) < 2.3"),
    filter = cms.bool(False)
	)

cleanedTaus = cms.EDProducer("TauValPFTauViewCleaner",
    srcObject            = cms.InputTag( "selectedTaus" ),
    srcObjectsToRemove   = cms.VInputTag( cms.InputTag("muons"), cms.InputTag("gsfElectrons") ),
    deltaRMin            = cms.double(0.15)
)

ZmtCandMuonTau = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("selectedMuonsIso@+ cleanedTaus@-"), # it takes opposite sign collection, no matter if +- or -+
    cut   = cms.string("5 < mass < 200")
	)

ZLegs  = cms.EDProducer(
    "CollectionFromZLegProducer", 
    ZCandidateCollection  = cms.InputTag("ZmtCandMuonTau"),      
	)


procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'RealTausData') #clones the sequence inside the process with RealTausData postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'RealTausData') #clones the sequence inside the process with RealTausData postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorRealTausData, 'kinematicSelectedTauValDenominator',  cms.InputTag("ZLegs","theProbeLeg")) #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence RealTausData to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'RealTausData')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorRealTausData.visit(zttModifier)

zttLabeler = lambda module : SetMassInput(module, cms.InputTag('ZLegs','mass'))
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorRealTausData.visit(zttModifier)


binning = cms.PSet(
    pt     = cms.PSet( nbins = cms.int32(10), min = cms.double(0.)   , max = cms.double(100.) ),
    eta    = cms.PSet( nbins = cms.int32(4) , min = cms.double(-3.)  , max = cms.double(3.)   ),
    phi    = cms.PSet( nbins = cms.int32(4) , min = cms.double(-180.), max = cms.double(180.) ),
    pileup = cms.PSet( nbins = cms.int32(18), min = cms.double(0.)   , max = cms.double(72.)  ),
    )
zttModifier = ApplyFunctionToSequence(lambda m: setBinning(m,binning))
proc.TauValNumeratorAndDenominatorRealTausData.visit(zttModifier)
#-----------------------------------------

#Sets the correct naming to efficiency histograms
proc.efficienciesRealTausData.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorRealTausData)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('RealTausData') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorRealTausData = cms.Sequence(
                            PrimaryVertexFilter * TauMETSelector *
                            ( selectedMuons * selectedMuonsIso + selectedTaus * cleanedTaus ) *   
                            ZmtCandMuonTau *
                            ZLegs
                            )

produceDenominator = produceDenominatorRealTausData

runTauValidationBatchMode = cms.Sequence(
      produceDenominator
      +TauValNumeratorAndDenominatorRealTausData
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesRealTausData
      )
