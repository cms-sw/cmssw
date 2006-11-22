#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "Validation/RecoTrack/interface/SiStripTrackingRecHitsValid.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MultiTrackValidator);
DEFINE_ANOTHER_FWK_MODULE(SiStripTrackingRecHitsValid);

#include "Validation/RecoTrack/interface/TrackEfficiencySelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

typedef 
ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,TrackEfficiencySelector> > 
TrackSelectorForEfficiency ;
DEFINE_ANOTHER_FWK_MODULE( TrackSelectorForEfficiency );

typedef 
ObjectSelector<SingleElementCollectionSelector<reco::TrackCollection,TrackFakeRateSelector.h> > 
TrackSelectorForFakeRate ;
DEFINE_ANOTHER_FWK_MODULE( TrackSelectorForFakeRate );

typedef 
ObjectSelector<SingleElementCollectionSelector<TrackingParticleCollection,TPEfficiencySelector.h> > 
TPSelectorForEfficiency ;
DEFINE_ANOTHER_FWK_MODULE( TPSelectorForEfficiency );

typedef 
ObjectSelector<SingleElementCollectionSelector<TrackingParticleCollection,TPFakeRateSelector.h> > 
TPSelectorForFakeRate ;
DEFINE_ANOTHER_FWK_MODULE( TPSelectorForFakeRate );

