import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
hltLhcParametersDefinerForTP = LhcParametersDefinerForTP.clone()
hltLhcParametersDefinerForTP.ComponentName = cms.string('hltLhcParametersDefinerForTP')
hltLhcParametersDefinerForTP.beamSpot      = cms.untracked.InputTag('hltOnlineBeamSpot')

from Validation.RecoTrack.MultiTrackValidator_cfi import *
hltMultiTrackValidator = multiTrackValidator.clone()
hltMultiTrackValidator.dirName = cms.string('HLT/Tracking/ValidationWRTtp/')
hltMultiTrackValidator.label   = cms.VInputTag(
    cms.InputTag("hltPixelTracks"),
#    cms.InputTag("hltPFJetCtfWithMaterialTracks"),
    cms.InputTag("hltPFlowTrackSelectionHighPurity"),
#    cms.InputTag("hltIter1PFJetCtfWithMaterialTracks"),
#    cms.InputTag("hltIter1PFlowTrackSelectionHighPurityLoose"),
#    cms.InputTag("hltIter1PFlowTrackSelectionHighPurityTight"),
    cms.InputTag("hltIter1PFlowTrackSelectionHighPurity"),
#    cms.InputTag("hltIter1Merged"),
#    cms.InputTag("hltIter2PFJetCtfWithMaterialTracks"),
    cms.InputTag("hltIter2PFlowTrackSelectionHighPurity"),
#    cms.InputTag("hltIter2Merged"),
#    cms.InputTag("hltIter3PFJetCtfWithMaterialTracks"),
#    cms.InputTag("hltIter3PFlowTrackSelectionHighPurityLoose"),
#    cms.InputTag("hltIter3PFlowTrackSelectionHighPurityTight"),
    cms.InputTag("hltIter3PFlowTrackSelectionHighPurity"),
#    cms.InputTag("hltIter3Merged"),
#    cms.InputTag("hltIter4PFJetCtfWithMaterialTracks"),
#    cms.InputTag("hltIter4PFlowTrackSelectionHighPurity"),
    cms.InputTag("hltIter4Merged"),
)
hltMultiTrackValidator.beamSpot = cms.InputTag("hltOnlineBeamSpot")
hltMultiTrackValidator.ptMinTP  = cms.double( 0.4)
hltMultiTrackValidator.lipTP    = cms.double(35.0)
hltMultiTrackValidator.tipTP    = cms.double(70.0)
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.ptMin = cms.double( 0.4)
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.lip   = cms.double(35.0)
hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.tip   = cms.double(70.0)
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsEta  = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPhi  = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsPt   = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXR = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.histoProducerAlgoBlock.TpSelectorForEfficiencyVsVTXZ = hltMultiTrackValidator.histoProducerAlgoBlock.generalTpSelector.clone()
hltMultiTrackValidator.parametersDefiner = cms.string('hltLhcParametersDefinerForTP')

from RecoTracker.DeDx.dedxHarmonic2_cfi import *
hltDedxHarmonic2 = dedxHarmonic2.clone()
hltDedxHarmonic2.tracks                     = cms.InputTag("hltIter4Merged")
hltDedxHarmonic2.trajectoryTrackAssociation = cms.InputTag("hltIter4Merged")

###
### Principal::getByToken: Found zero products matching all criteria
### Looking for type: edm::AssociationMap<edm::OneToOne<std::vector<Trajectory>,std::vector<reco::Track>,unsigned short> >
### Looking for module label: hltIter4Merged
#hltMultiTrackValidator.dEdx1Tag = cms.InputTag("hltDedxHarmonic2")

from RecoTracker.DeDx.dedxTruncated40_cfi import *
hltDedxTruncated40 = dedxTruncated40.clone()
hltDedxTruncated40.tracks                     = cms.InputTag("hltIter4Merged")
hltDedxTruncated40.trajectoryTrackAssociation = cms.InputTag("hltIter4Merged")

#hltMultiTrackValidator.dEdx2Tag = cms.InputTag("hltDedxTruncated40")

hltQuickTrackAssociatorByHits = cms.ESProducer("QuickTrackAssociatorByHitsESProducer",
    Quality_SimToReco = cms.double(0.5),
    cluster2TPSrc = cms.InputTag("hltTPClusterProducer"),
    associatePixel = cms.bool(True),
    useClusterTPAssociation = cms.bool(False),
    Purity_SimToReco = cms.double(0.75),
    ThreeHitTracksAreSpecial = cms.bool(True),
    AbsoluteNumberOfHits = cms.bool(False),
    associateStrip = cms.bool(True),
    Cut_RecoToSim = cms.double(0.75),
    SimToRecoDenominator = cms.string('sim'),
    ComponentName = cms.string('hltQuickTrackAssociatorByHits')
)

hltTrackingParticleRecoTrackAsssociation = cms.EDProducer("TrackAssociatorEDProducer",
    label_tr = cms.InputTag("hltIter4Merged"),
    associator = cms.string('hltQuickTrackAssociatorByHits'),
    label_tp = cms.InputTag("mix","MergedTrackTruth"),
    ignoremissingtrackcollection = cms.untracked.bool(False)
)

hltMultiTrackValidator.associatormap = cms.InputTag("hltTrackingParticleRecoTrackAsssociation")


hltTPClusterProducer = cms.EDProducer("ClusterTPAssociationProducer",
    stripSimLinkSrc = cms.InputTag("simSiStripDigis"),
    verbose = cms.bool(False),
    pixelClusterSrc = cms.InputTag("hltSiPixelClusters"),
    pixelSimLinkSrc = cms.InputTag("simSiPixelDigis"),
    trackingParticleSrc = cms.InputTag("mix","MergedTrackTruth"),
#    stripClusterSrc = cms.InputTag("hltSiStripClusters"),
    stripClusterSrc = cms.InputTag("hltSiStripRawToClustersFacility"),                                      
    simTrackSrc = cms.InputTag("g4SimHits")
)
hltTrackAssociatorByHitsRecoDenom = cms.ESProducer("QuickTrackAssociatorByHitsESProducer",
    Quality_SimToReco = cms.double(0.5),
    associatePixel = cms.bool(True),
    useClusterTPAssociation = cms.bool(True),
    Purity_SimToReco = cms.double(0.75),
    Cut_RecoToSim = cms.double(0.75),
    ThreeHitTracksAreSpecial = cms.bool(True),
    AbsoluteNumberOfHits = cms.bool(False),
    associateStrip = cms.bool(True),
    ComponentName = cms.string('hltTrackAssociatorByHitsRecoDenom'),
    SimToRecoDenominator = cms.string('reco'),
    cluster2TPSrc = cms.InputTag("hltTPClusterProducer")
)
hltMultiTrackValidator.associators = cms.vstring('hltTrackAssociatorByHitsRecoDenom')

from Validation.RecoTrack.cutsTPEffic_cfi import *
from Validation.RecoTrack.cutsTPFake_cfi import *

from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import *

hltMultiTrackValidation = cms.Sequence(
#    hltDedxHarmonic2
#    + hltDedxTruncated40
    simHitTPAssocProducer
    + hltTPClusterProducer
    + hltTrackingParticleRecoTrackAsssociation
    + cutsTPEffic
    + cutsTPFake
    + hltMultiTrackValidator
)    

