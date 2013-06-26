#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "Validation/RecoTrack/interface/MultiTrackValidatorGenPs.h"
#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoFactory.h"
#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoForTracker.h"

#include "Validation/RecoTrack/interface/TrackerSeedValidator.h"
#include "Validation/RecoTrack/interface/SiStripTrackingRecHitsValid.h"
#include "Validation/RecoTrack/interface/SiPixelTrackingRecHitsValid.h"

DEFINE_FWK_MODULE(MultiTrackValidator);
DEFINE_FWK_MODULE(MultiTrackValidatorGenPs);
DEFINE_FWK_MODULE(TrackerSeedValidator);
DEFINE_FWK_MODULE(SiStripTrackingRecHitsValid);
DEFINE_FWK_MODULE(SiPixelTrackingRecHitsValid);

DEFINE_EDM_PLUGIN(MTVHistoProducerAlgoFactory, MTVHistoProducerAlgoForTracker,  "MTVHistoProducerAlgoForTracker");


// #include "Validation/RecoTrack/interface/RecoTrackSelector.h"
// #include "Validation/RecoTrack/interface/TrackEfficiencySelector.h"
// #include "Validation/RecoTrack/interface/TPEfficiencySelector.h"
// #include "Validation/RecoTrack/interface/TrackFakeRateSelector.h"
// #include "Validation/RecoTrack/interface/TPFakeRateSelector.h"
// #include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
// #include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

// typedef 
// ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,RecoTrackSelector> > 
// TrackSelector ;
// DEFINE_FWK_MODULE( TrackSelector );

// typedef 
// ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,TrackEfficiencySelector> > 
// TrackSelectorForEfficiency ;
// DEFINE_FWK_MODULE( TrackSelectorForEfficiency );

// typedef 
// ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,TrackFakeRateSelector> > 
// TrackSelectorForFakeRate ;
// DEFINE_FWK_MODULE( TrackSelectorForFakeRate );

// typedef 
// ObjectSelector<SingleElementCollectionSelector<TrackingParticleCollection,TPEfficiencySelector> > 
// TPSelectorForEfficiency ;
// DEFINE_FWK_MODULE( TPSelectorForEfficiency );

// typedef 
// ObjectSelector<SingleElementCollectionSelector<TrackingParticleCollection,TPFakeRateSelector> > 
// TPSelectorForFakeRate ;
// DEFINE_FWK_MODULE( TPSelectorForFakeRate );

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

typedef SingleObjectSelector<
            std::vector<reco::Track>,
            StringCutObjectSelector<reco::Track>
        > TrackSelectorForValidation;

DEFINE_FWK_MODULE(TrackSelectorForValidation);
