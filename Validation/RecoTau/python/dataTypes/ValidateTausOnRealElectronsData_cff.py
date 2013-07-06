import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from RecoJets.Configuration.RecoPFJets_cff import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

ElPrimaryVertexFilter = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
    filter = cms.bool(False)
    )

ElBestPV = cms.EDProducer( 
    "HighestSumP4PrimaryVertexSelector",
    src = cms.InputTag("ElPrimaryVertexFilter")
	)

selectedElectrons = cms.EDFilter(
    "TauValElectronSelector",
    src = cms.InputTag('gsfElectrons'),
    cut = cms.string("pt > 25.0 && abs(eta) < 2.4 && isElectron"),
    filter = cms.bool(False)
	)

ElectronsFromPV = cms.EDProducer(
    "GsfElectronFromPVSelector",
    srcElectron = cms.InputTag("selectedElectrons"),
    srcVertex   = cms.InputTag("ElBestPV"),
    max_dxy     = cms.double(0.01),
    max_dz      = cms.double(0.1)
    )

idElectrons = cms.EDFilter(
    "TauValElectronSelector",
    src = cms.InputTag('ElectronsFromPV'),
    cut = cms.string('ecalDrivenSeed & isGsfCtfScPixChargeConsistent & isGsfScPixChargeConsistent & isGsfCtfChargeConsistent & !isEBEEGap & (isEB & sigmaIetaIeta<0.01 & abs(deltaPhiSuperClusterTrackAtVtx)<0.06 & abs(deltaEtaSuperClusterTrackAtVtx)<0.006 & hadronicOverEm<0.04 | isEE & sigmaIetaIeta<0.03 & abs(deltaPhiSuperClusterTrackAtVtx)<0.04 & abs(deltaEtaSuperClusterTrackAtVtx)<0.007 & hadronicOverEm<0.025)'),
    filter = cms.bool(False)
)

trackElectrons = cms.EDFilter(
    "TauValElectronSelector",
    src = cms.InputTag('idElectrons'),
    cut = cms.string('gsfTrack.isNonnull  && 0.7 < eSuperClusterOverP < 1.5'),
#    cut = cms.string('gsfTrack.isNonnull && gsfTrack.trackerExpectedHitsInner.numberOfHits = 0 && 0.7 < eSuperClusterOverP < 1.5'),
    filter = cms.bool(False)
)

isolatedElectrons = cms.EDFilter(
    "TauValElectronSelector",
    src = cms.InputTag('trackElectrons'),
    cut = cms.string("(isEB & (dr04TkSumPt/pt + max(0.,dr04EcalRecHitSumEt-2.)/pt + dr04HcalTowerSumEt/pt < 0.10)) | (isEE & (dr04TkSumPt/pt + dr04EcalRecHitSumEt/pt + dr04HcalTowerSumEt/pt < 0.09))"),
    filter = cms.bool(False)
	)

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

ElGoodTracks = cms.EDFilter(
    "TrackSelector",
    src = cms.InputTag("generalTracks"), 
    cut = cms.string("pt > 5 && abs(eta) < 2.5"),
    filter = cms.bool(False)
	)

ElIsoTracks = cms.EDProducer(
    "IsoTracks",
    src           = cms.InputTag("ElGoodTracks"),
    radius        = cms.double(0.3),
    SumPtFraction = cms.double(0.5)
    )

ElTrackFromPV = cms.EDProducer(
    "TrackFromPVSelector",
    srcTrack   = cms.InputTag("ElIsoTracks"),
    srcVertex  = cms.InputTag("ElBestPV"),
    max_dxy    = cms.double(0.01),
    max_dz     = cms.double(0.1)
    )

ElTrackCands  = cms.EDProducer(
    "ConcreteChargedCandidateProducer", 
    src  = cms.InputTag("ElTrackFromPV"),      
    particleType = cms.string("e+")     # this is needed to define a mass do not trust the sign, it is dummy
	)

ZeeCandElectronTrack = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("isolatedElectrons@+ ElTrackCands@-"), # it takes opposite sign collection, no matter if +- or -+
    cut   = cms.string("80 < mass < 100")
	)

BestZee = cms.EDProducer(
    "BestMassZArbitrationProducer", # returns the Z with mass closest to 91.18 GeV
	ZCandidateCollection = cms.InputTag("ZeeCandElectronTrack")
	)

ElZLegs  = cms.EDProducer(
    "CollectionFromZLegProducer", 
    ZCandidateCollection  = cms.InputTag("BestZee"),      
	)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'RealElectronsData') #clones the sequence inside the process with RealElectronsData postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'RealElectronsData') #clones the sequence inside the process with RealElectronsData postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorRealElectronsData, 'kinematicSelectedTauValDenominator', cms.InputTag("ElZLegs","theProbeLeg")) #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence RealElectronsData to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'RealElectronsData')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorRealElectronsData.visit(zttModifier)

binning = cms.PSet(
    pt = cms.PSet( nbins = cms.int32(10), min = cms.double(0.), max = cms.double(100.) ), #hinfo(75, 0., 150.)
    eta = cms.PSet( nbins = cms.int32(4), min = cms.double(-3.), max = cms.double(3.) ), #hinfo(60, -3.0, 3.0);
    phi = cms.PSet( nbins = cms.int32(4), min = cms.double(-180.), max = cms.double(180.) ), #hinfo(36, -180., 180.);
    pileup = cms.PSet( nbins = cms.int32(18), min = cms.double(0.), max = cms.double(72.) ),#hinfo(25, 0., 25.0);
    )
zttModifier = ApplyFunctionToSequence(lambda m: setBinning(m,binning))
proc.TauValNumeratorAndDenominatorRealElectronsData.visit(zttModifier)
#-----------------------------------------

#Sets the correct naming to efficiency histograms
proc.efficienciesRealElectronsData.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorRealElectronsData)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = filter( lambda x: (x not in procAttributes) and (x.find('RealElectronsData') != -1), dir(proc) )

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorRealElectronsData = cms.Sequence( ElPrimaryVertexFilter * ElBestPV *
                                                    ( (selectedElectrons * ElectronsFromPV * idElectrons * trackElectrons * isolatedElectrons) +
                                                      (ElGoodTracks * ElIsoTracks * ElTrackFromPV * ElTrackCands) ) *
                                                    ZeeCandElectronTrack *
                                                    BestZee *
                                                    ElZLegs 
                                                  )

produceDenominator = cms.Sequence(produceDenominatorRealElectronsData)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorRealElectronsData
      +TauValNumeratorAndDenominatorRealElectronsData
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesRealElectronsData
      )
