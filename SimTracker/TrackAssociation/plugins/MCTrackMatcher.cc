/** \class MCTrackMatcher 
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace edm { class ParameterSet; }

class MCTrackMatcher : public edm::EDProducer {
 public:
  /// constructor
  MCTrackMatcher( const edm::ParameterSet & );

 private:
  void produce( edm::Event& evt, const edm::EventSetup& es ) override;
  edm::InputTag associator_;
  edm::InputTag tracks_, genParticles_, trackingParticles_;
  typedef edm::Association<reco::GenParticleCollection> GenParticleMatch;
};

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
using namespace edm;
using namespace std;
using namespace reco;

MCTrackMatcher::MCTrackMatcher(const ParameterSet & p) :
  associator_(p.getParameter<string>("associator")),
  tracks_(p.getParameter<InputTag>("tracks")),
  genParticles_( p.getParameter<InputTag>("genParticles")),
  trackingParticles_( p.getParameter<InputTag>("trackingParticles")) {
  produces<GenParticleMatch>();
  
  consumes<reco::TrackToTrackingParticleAssociator>(associator_);
  consumes<edm::View<reco::Track>>(tracks_);
  consumes<GenParticleCollection>(genParticles_);
  consumes<TrackingParticleCollection>(trackingParticles_);
}

void MCTrackMatcher::produce(Event& evt, const EventSetup& es) {
  Handle<reco::TrackToTrackingParticleAssociator> assoc;  
  evt.getByLabel(associator_,assoc);
  const reco::TrackToTrackingParticleAssociator * associator = assoc.product();
  Handle<View<Track> > tracks;
  evt.getByLabel(tracks_, tracks);
  Handle<TrackingParticleCollection> trackingParticles;
  evt.getByLabel(trackingParticles_,trackingParticles);
  Handle<vector<int> > barCodes;
  evt.getByLabel(genParticles_,barCodes );
  Handle<GenParticleCollection> genParticles;
  evt.getByLabel(genParticles_, genParticles );
  RecoToSimCollection associations = associator->associateRecoToSim ( tracks, trackingParticles);
  auto_ptr<GenParticleMatch> match(new GenParticleMatch(GenParticleRefProd(genParticles)));
  GenParticleMatch::Filler filler(*match);
  size_t n = tracks->size();
  vector<int> indices(n,-1);
  for (size_t i = 0; i < n; ++ i ) {
    RefToBase<Track> track(tracks, i);
    RecoToSimCollection::const_iterator f = associations.find(track);
    if ( f != associations.end() ) {
      TrackingParticleRef tp = f->val.front().first;
      TrackingParticle::genp_iterator j, b = tp->genParticle_begin(), e = tp->genParticle_end();
      for( j = b; j != e; ++ j ) {
	const reco::GenParticle * p = j->get();
	if (p->status() == 1) {
	  indices[i] = j->key();
	  break;
	}
      }
    }
  }
  filler.insert(tracks, indices.begin(), indices.end());
  filler.fill();
  evt.put(match);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTrackMatcher );

