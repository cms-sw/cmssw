#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "Validation/RecoTrack/interface/SiStripTrackingRecHitsValid.h"
#include "Validation/RecoTrack/interface/SiPixelTrackingRecHitsValid.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MultiTrackValidator);
DEFINE_ANOTHER_FWK_MODULE(SiStripTrackingRecHitsValid);
DEFINE_ANOTHER_FWK_MODULE(SiPixelTrackingRecHitsValid);

// #include "Validation/RecoTrack/interface/RecoTrackSelector.h"
// #include "Validation/RecoTrack/interface/TrackEfficiencySelector.h"
// #include "Validation/RecoTrack/interface/TPEfficiencySelector.h"
// #include "Validation/RecoTrack/interface/TrackFakeRateSelector.h"
// #include "Validation/RecoTrack/interface/TPFakeRateSelector.h"
// #include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
// #include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

// typedef 
// ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,RecoTrackSelector> > 
// TrackSelector ;
// DEFINE_ANOTHER_FWK_MODULE( TrackSelector );

// typedef 
// ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,TrackEfficiencySelector> > 
// TrackSelectorForEfficiency ;
// DEFINE_ANOTHER_FWK_MODULE( TrackSelectorForEfficiency );

// typedef 
// ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,TrackFakeRateSelector> > 
// TrackSelectorForFakeRate ;
// DEFINE_ANOTHER_FWK_MODULE( TrackSelectorForFakeRate );

// typedef 
// ObjectSelector<SingleElementCollectionSelector<TrackingParticleCollection,TPEfficiencySelector> > 
// TPSelectorForEfficiency ;
// DEFINE_ANOTHER_FWK_MODULE( TPSelectorForEfficiency );

// typedef 
// ObjectSelector<SingleElementCollectionSelector<TrackingParticleCollection,TPFakeRateSelector> > 
// TPSelectorForFakeRate ;
// DEFINE_ANOTHER_FWK_MODULE( TPSelectorForFakeRate );


