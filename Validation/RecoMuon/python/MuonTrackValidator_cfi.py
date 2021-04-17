import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.histoParameters_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
muonTrackValidator = DQMEDAnalyzer("MuonTrackValidator",
    # define the TrackingParticleSelector for evaluation of efficiency
    muonTPSelector = cms.PSet(muonTPSet),
    # input TrackingParticle collections
    label_tp_effic = cms.InputTag("mix","MergedTrackTruth"),
    label_tp_fake = cms.InputTag("mix","MergedTrackTruth"),
    label_pileupinfo = cms.InputTag("addPileupInfo"),
    #
    # input reco::Track collection
    label = cms.VInputTag(cms.InputTag("globalMuons")),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    #
    # set true if you do not want that MTV launch an exception
    # if the track collection is missing (e.g. HLT):
    ignoremissingtrackcollection=cms.untracked.bool(False),
    #
    # collision-like tracks
    parametersDefiner = cms.string('LhcParametersDefinerForTP'),
    # cosmics tracks
    # parametersDefiner = cms.string('CosmicParametersDefinerForTP'), 
    #
    # map linking SimHits to TrackingParticles, needed for cosmics validation`
    simHitTpMapTag = cms.InputTag("simHitTPAssocProducer"), 
    #
    # if !UseAssociators the association map has to be given in input 
    associators = cms.vstring('MuonAssociationByHits'),
    UseAssociators = cms.bool(False),
    useGEMs = cms.bool(False),
    useME0 = cms.bool(False),
    associatormap = cms.InputTag("tpToMuonTrackAssociation"),
    #
    # BiDirectional Logic for RecoToSim association corrects the Fake rates (counting ghosts and split tracks as fakes)
    #  setting it to False the ghost and split tracks are counted as good ones
    #  the default setting is True: should NOT be changed !
    BiDirectional_RecoToSim_association = cms.bool(True),
    #
    # Output File / Directory
    outputFile = cms.string(''),
    dirName = cms.string('Muons/RecoMuonV/MuonTrack/'),
    #
    # Parameters defining which histograms to make and their attributes (nbins, range: min, max...)
    muonHistoParameters = cms.PSet(defaultMuonHistoParameters)
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( muonTrackValidator, useGEMs = cms.bool(True) )
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify( muonTrackValidator, useME0 = cms.bool(True) )
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0
phase2_GE0.toModify( muonTrackValidator, useME0 = cms.bool(False) )

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(muonTrackValidator,
    label_tp_effic = "mixData:MergedTrackTruth",
    label_tp_fake = "mixData:MergedTrackTruth",
)
