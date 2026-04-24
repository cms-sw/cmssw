# Validation for the superclustering sequence in TICL
# monitors efficiency of the PID cut and of supercluster building (in trackster dataformat)
# to be used on samples containing prompt electrons (or photons)
import FWCore.ParameterSet.Config as cms

from Validation.HGCalValidation.tracksterAssociationMaskProducer_cfi import tracksterAssociationMaskProducer as _tracksterAssociationMaskProducer
from Validation.HGCalValidation.tracksterSuperclusteringValidCandidateMaskProducer_cfi import tracksterSuperclusteringValidCandidateMaskProducer as _tracksterSuperclusteringValidCandidateMaskProducer


sourceTracksterIteration = "ticlTrackstersCLUE3DHigh" # trackster collection used as input to superclustering (CLUE3D tracksters)
superclusterTracksterIteration = "ticlTracksterLinksSuperclusteringDNN" # trackster collection output by superclustering (superclusters in trackster dataformat)
simTrackstersCollection = "ticlSimTrackstersfromCPs" # simtrackster collection used for genmatching (CaloParticle)
sim2recoscore = 0.3 # cut on sim2reco score to consider a supercluster genmatched


################ Validation of PID cut for superclustering

## Building masks to select tracksters for efficiency and fake rate "baseline"
# finding the reco trackster that is the "seed" cluster for each electron caloparticle
ticlValidSuperclusteringSeedMask = _tracksterAssociationMaskProducer.clone(
    tracksters=cms.InputTag(sourceTracksterIteration),
    associatorRecoToSim=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{sourceTracksterIteration}To{simTrackstersCollection}"),
    associatorSimToReco=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{simTrackstersCollection}To{sourceTracksterIteration}"),
    recoToSimScoreCut=cms.double(0.1), # for the seed cluster of an electron, we want a tight cut (will remove low energy electron clusters that are heavily pileup contaminated)
    recoToSimScoreCut_forFakes=cms.double(0.8), # we don't want to study fake rates of PID in tracksters that have some electron/photon in it
    particleTypesSignal=cms.vint32(0, 1) # ele+photon (SimTrackster ParticleType enum, photon=0, ele=1, muon=2, pi0=3, h+-=4, h0=5)
)

# finding all reco tracksters that are a "brem"/"superclustering candidate" cluster of an electron
tracksterSuperclusteringValidCandidateMaskProducer = _tracksterSuperclusteringValidCandidateMaskProducer.clone(
    tracksters=cms.InputTag(sourceTracksterIteration),
    associatorRecoToSim=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{sourceTracksterIteration}To{simTrackstersCollection}"),
    associatorSimToReco=cms.InputTag(f"allTrackstersToSimTrackstersAssociationsByLCs:{simTrackstersCollection}To{sourceTracksterIteration}"),
    recoToSimScoreCut=cms.double(0.15), # for the candidate brem clusters of an electron, we want a slightly looser cut
    particleTypesSignal=cms.vint32(0, 1) # ele+photon (SimTrackster ParticleType enum, photon=0, ele=1, muon=2, pi0=3, h+-=4, h0=5)
)

## actual validation DQM
from Validation.HGCalValidation.ticlTracksterPIDValidation_cfi import ticlTracksterPIDValidation as _ticlTracksterPIDValidation
ticlValidSuperclusteringSeedPID = _ticlTracksterPIDValidation.clone(
    tracksters = cms.InputTag(sourceTracksterIteration),
    tracksterMask = cms.InputTag("ticlValidSuperclusteringSeedMask"),
    tracksterMaskFakes = cms.InputTag("ticlValidSuperclusteringSeedMask", "fakes"),
    folder = cms.string("HGCAL/TICLTracksterPIDValidation/superclusteringSeedTrackster/"),
    pidCut = cms.double(0.5)
)
ticlValidSuperclusteringCandidatePID = _ticlTracksterPIDValidation.clone(
    tracksters = cms.InputTag(sourceTracksterIteration),
    tracksterMask = cms.InputTag("tracksterSuperclusteringValidCandidateMaskProducer"),
    # for fakes the selection is the same as for seeds, only the PID cut is potentially different
    tracksterMaskFakes = cms.InputTag("ticlValidSuperclusteringSeedMask", "fakes"),
    folder = cms.string("HGCAL/TICLTracksterPIDValidation/superclusteringCandidateTrackster/"),
    pidCut = cms.double(0.2),
)

ticlSuperclusterPIDValidation = cms.Sequence(
    ticlValidSuperclusteringSeedMask + tracksterSuperclusteringValidCandidateMaskProducer +
    ticlValidSuperclusteringSeedPID + ticlValidSuperclusteringCandidatePID
)



#### post-processing : computing efficiencies of PID cut as ratios of histogram
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
postProcessorTICLPIDValid = DQMEDHarvester('DQMGenericClient',
    subDirs = cms.untracked.vstring("HGCAL/TICLTracksterPIDValidation/superclusteringSeedTrackster", "HGCAL/TICLTracksterPIDValidation/superclusteringCandidateTrackster"),
    efficiencySets = cms.untracked.VPSet(
        cms.untracked.PSet( # validating the efficiency of the gen-matching selections on electron caloparticles
            name=cms.untracked.string("pt_eta_reco2SimSelection_eff"),
            title=cms.untracked.string("Efficiency of reco2Sim cut as a function of pt and abs(eta) on superclusteringSeedTrackster"),
            numerator=cms.untracked.string("pt_eta_reco2SimSelected"),
            denominator=cms.untracked.string("pt_eta_noReco2SimSelection")
        ),

        cms.untracked.PSet(
            name=cms.untracked.string("pt_eta_PID_eff"),
            title=cms.untracked.string("Efficiency of PID cut as a function of pt and abs(eta) on superclusteringSeedTrackster"),
            numerator=cms.untracked.string("pt_eta_pidNum"),
            denominator=cms.untracked.string("pt_eta_reco2SimSelected")
        ),
        cms.untracked.PSet(
            name=cms.untracked.string("pt_eta_PID_fakeRate"),
            title=cms.untracked.string("Fake rate of PID cut as a function of pt and abs(eta) on superclusteringSeedTrackster (computed on tracksters failing sim-matching to electron)"),
            numerator=cms.untracked.string("pt_eta_fakes_pid_Num"),
            denominator=cms.untracked.string("pt_eta_fakes")
        ),

        
    ),
    efficiency = cms.vstring(),
    resolution = cms.vstring(),
    verbose = cms.untracked.uint32(4))


########################### Sequences
ticlSuperclusterValidation = cms.Sequence(
    ticlSuperclusterPIDValidation
)
postProcessorTiclSupercluster = cms.Sequence(
    postProcessorTICLPIDValid
)
