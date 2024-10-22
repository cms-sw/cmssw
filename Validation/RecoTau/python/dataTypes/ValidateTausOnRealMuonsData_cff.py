import FWCore.ParameterSet.Config as cms
from Validation.RecoTau.RecoTauValidation_cfi import *
import copy

from RecoJets.Configuration.RecoPFJets_cff import *
import PhysicsTools.PatAlgos.tools.helpers as helpers

MuPrimaryVertexFilter = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
    filter = cms.bool(False)
    )

MuBestPV = cms.EDProducer(
    "HighestSumP4PrimaryVertexSelector",
    src = cms.InputTag("MuPrimaryVertexFilter")
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
    cut = cms.string('(isolationR03().emEt + isolationR03().hadEt + isolationR03().sumPt)/pt < 0.15'),
    filter = cms.bool(False)
	)    

MuonsFromPV = cms.EDProducer(
    "MuonFromPVSelector",
    srcMuon    = cms.InputTag("selectedMuonsIso"),
    srcVertex  = cms.InputTag("MuBestPV"),
    max_dxy    = cms.double(0.01),
    max_dz     = cms.double(0.1)
    )

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

MuGoodTracks = cms.EDFilter("TrackSelector",
    src = cms.InputTag("generalTracks"), 
    cut = cms.string("pt > 5 && abs(eta) < 2.5"),
    filter = cms.bool(False)
	)

MuIsoTracks = cms.EDProducer(
    "IsoTracks",
    src           = cms.InputTag("MuGoodTracks"),
    radius        = cms.double(0.3),
    SumPtFraction = cms.double(0.5)
    )

MuTrackFromPV = cms.EDProducer(
    "TrackFromPVSelector",
    srcTrack   = cms.InputTag("MuIsoTracks"),
    srcVertex  = cms.InputTag("MuBestPV"),
    max_dxy    = cms.double(0.01),
    max_dz     = cms.double(0.1)
    )

MuTrackCands  = cms.EDProducer(
    "ConcreteChargedCandidateProducer", 
    src  = cms.InputTag("MuTrackFromPV"),      
    particleType = cms.string("mu+")     # this is needed to define a mass
	)

ZmmCandMuonTrack = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("MuonsFromPV@+ MuTrackCands@-"), # it takes opposite sign collection, no matter if +- or -+
    cut   = cms.string("80 < mass < 100")
	)

BestZmm = cms.EDProducer("BestMassZArbitrationProducer", # returns the Z with mass closer to 91.18 GeV
	ZCandidateCollection = cms.InputTag("ZmmCandMuonTrack")
	)

MuZLegs  = cms.EDProducer("CollectionFromZLegProducer", 
    ZCandidateCollection  = cms.InputTag("BestZmm"),      
	)

procAttributes = dir(proc) #Takes a snapshot of what there in the process
helpers.cloneProcessingSnippet( proc, proc.TauValNumeratorAndDenominator, 'RealMuonsData') #clones the sequence inside the process with RealMuonsData postfix
helpers.cloneProcessingSnippet( proc, proc.TauEfficiencies, 'RealMuonsData') #clones the sequence inside the process with RealMuonsData postfix
helpers.massSearchReplaceAnyInputTag(proc.TauValNumeratorAndDenominatorRealMuonsData, 'kinematicSelectedTauValDenominator',  cms.InputTag("MuZLegs","theProbeLeg")) #sets the correct input tag

#adds to TauValNumeratorAndDenominator modules in the sequence RealMuonsData to the extention name
zttLabeler = lambda module : SetValidationExtention(module, 'RealMuonsData')
zttModifier = ApplyFunctionToSequence(zttLabeler)
proc.TauValNumeratorAndDenominatorRealMuonsData.visit(zttModifier)

binning = cms.PSet(
    pt = cms.PSet( nbins = cms.int32(10), min = cms.double(0.), max = cms.double(100.) ), #hinfo(75, 0., 150.)
    eta = cms.PSet( nbins = cms.int32(4), min = cms.double(-3.), max = cms.double(3.) ), #hinfo(60, -3.0, 3.0);
    phi = cms.PSet( nbins = cms.int32(4), min = cms.double(-180.), max = cms.double(180.) ), #hinfo(36, -180., 180.);
    pileup = cms.PSet( nbins = cms.int32(18), min = cms.double(0.), max = cms.double(72.) ),#hinfo(25, 0., 25.0);
    )
zttModifier = ApplyFunctionToSequence(lambda m: setBinning(m,binning))
proc.TauValNumeratorAndDenominatorRealMuonsData.visit(zttModifier)
#-----------------------------------------

#Set discriminators
discs_to_retain = ['ByDecayModeFinding', 'MuonRejection']
proc.RunHPSValidationRealMuonsData.discriminators = cms.VPSet([p for p in proc.RunHPSValidationRealMuonsData.discriminators if any(disc in p.discriminator.value() for disc in discs_to_retain) ])

#Sets the correct naming to efficiency histograms
proc.efficienciesRealMuonsData.plots = Utils.SetPlotSequence(proc.TauValNumeratorAndDenominatorRealMuonsData)
proc.efficienciesRealMuonsDataSummary = cms.EDProducer("TauDQMHistEffProducer",
    plots = cms.PSet(
        Summary = cms.PSet(
            denominator = cms.string('RecoTauV/standardValidation/hpsPFTauProducerRealMuonsData_Summary/#PAR#PlotDen'),
            efficiency = cms.string('RecoTauV/standardValidation/hpsPFTauProducerRealMuonsData_Summary/#PAR#Plot'),
            numerator = cms.string('RecoTauV/standardValidation/hpsPFTauProducerRealMuonsData_Summary/#PAR#PlotNum'),
            parameter = cms.vstring('summary'),
            stepByStep = cms.bool(True)
        ),
    )
)

#checks what's new in the process (the cloned sequences and modules in them)
newProcAttributes = [x for x in dir(proc) if (x not in procAttributes) and (x.find('RealMuonsData') != -1)]

#spawns a local variable with the same name as the proc attribute, needed for future process.load
for newAttr in newProcAttributes:
    locals()[newAttr] = getattr(proc,newAttr)

produceDenominatorRealMuonsData = cms.Sequence(
                            cms.ignore(MuPrimaryVertexFilter) * MuBestPV *
                            ( ( cms.ignore(selectedMuons) * cms.ignore(selectedMuonsIso) * MuonsFromPV ) +
                              ( cms.ignore(MuGoodTracks) * MuIsoTracks * MuTrackFromPV * MuTrackCands ) ) *
                            ZmmCandMuonTrack *
                            BestZmm *
                            MuZLegs 
                            )

produceDenominator = cms.Sequence(produceDenominatorRealMuonsData)

runTauValidationBatchMode = cms.Sequence(
      produceDenominatorRealMuonsData
      +TauValNumeratorAndDenominatorRealMuonsData
      )

runTauValidation = cms.Sequence(
      runTauValidationBatchMode
      +TauEfficienciesRealMuonsData
      )
