#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "SimTracker/TrackHistory/plugins/TrackParticleSelector.h"

#define DEFINE_TRACKPARTICLE_SELECTOR(Collection, Category) \
typedef ObjectSelector<TrackParticleSelector<Collection, TrackCategories::Category> > Category##Collection##Selector; \
DEFINE_FWK_MODULE( Category##Collection##Selector )


namespace reco{
  namespace modules {
	
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, SignalEvent);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, PV);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, SV);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, TV);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Displaced);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Ks);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Lambda);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, PhotonConversion);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Light);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Charm);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Bottom);

    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Fake);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Bad);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, SignalEvent);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, PV);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, SV);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, TV);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Displaced);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Ks);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Lambda);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, PhotonConversion);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Light);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Charm);
    DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Bottom);

  }
}
