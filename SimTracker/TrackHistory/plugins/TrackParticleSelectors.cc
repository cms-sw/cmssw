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


namespace reco
{
namespace modules
{

DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, SignalEvent);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Bottom);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Charm);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Light);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, BWeakDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, CWeakDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, TauDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, KsDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, LambdaDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, LongLivedDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Conversion);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Interaction);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, PrimaryVertex);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, SecondaryVertex);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, TierciaryVertex);
DEFINE_TRACKPARTICLE_SELECTOR(TrackingParticleCollection, Unknown);

DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Fake);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Bad);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, SignalEvent);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Bottom);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Charm);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Light);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, BWeakDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, CWeakDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, TauDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, KsDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, LambdaDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, LongLivedDecay);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Conversion);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Interaction);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, PrimaryVertex);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, SecondaryVertex);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, TierciaryVertex);
DEFINE_TRACKPARTICLE_SELECTOR(TrackCollection, Unknown);

}
}
