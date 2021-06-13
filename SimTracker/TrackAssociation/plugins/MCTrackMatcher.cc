/** \class MCTrackMatcher
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

namespace edm {
  class ParameterSet;
}

using namespace edm;
using namespace std;
using namespace reco;

class MCTrackMatcher : public edm::global::EDProducer<> {
public:
  /// constructor
  MCTrackMatcher(const edm::ParameterSet &);

private:
  void produce(edm::StreamID, edm::Event &evt, const edm::EventSetup &es) const override;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> associator_;
  edm::EDGetTokenT<edm::View<reco::Track>> tracks_;
  edm::EDGetTokenT<GenParticleCollection> genParticles_;
  edm::EDGetTokenT<std::vector<int>> genParticleInts_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticles_;
  bool throwOnMissingTPCollection_;
  typedef edm::Association<reco::GenParticleCollection> GenParticleMatch;
};

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

MCTrackMatcher::MCTrackMatcher(const ParameterSet &p)
    : associator_(consumes<reco::TrackToTrackingParticleAssociator>(p.getParameter<string>("associator"))),
      tracks_(consumes<edm::View<reco::Track>>(p.getParameter<InputTag>("tracks"))),
      genParticles_(consumes<GenParticleCollection>(p.getParameter<InputTag>("genParticles"))),
      genParticleInts_(consumes<std::vector<int>>(p.getParameter<InputTag>("genParticles"))),
      trackingParticles_(consumes<TrackingParticleCollection>(p.getParameter<InputTag>("trackingParticles"))),
      throwOnMissingTPCollection_(p.getParameter<bool>("throwOnMissingTPCollection")) {
  produces<GenParticleMatch>();
}

void MCTrackMatcher::produce(edm::StreamID, Event &evt, const EventSetup &es) const {
  Handle<reco::TrackToTrackingParticleAssociator> assoc;
  evt.getByToken(associator_, assoc);
  const reco::TrackToTrackingParticleAssociator *associator = assoc.product();
  Handle<View<Track>> tracks;
  evt.getByToken(tracks_, tracks);
  Handle<TrackingParticleCollection> trackingParticles;
  auto trackingParticlesFound = evt.getByToken(trackingParticles_, trackingParticles);
  Handle<vector<int>> barCodes;
  evt.getByToken(genParticleInts_, barCodes);
  Handle<GenParticleCollection> genParticles;
  evt.getByToken(genParticles_, genParticles);
  unique_ptr<GenParticleMatch> match(new GenParticleMatch(GenParticleRefProd(genParticles)));
  if (not throwOnMissingTPCollection_ and not trackingParticlesFound) {
    evt.put(std::move(match));
    return;
  }
  RecoToSimCollection associations = associator->associateRecoToSim(tracks, trackingParticles);
  GenParticleMatch::Filler filler(*match);
  size_t n = tracks->size();
  vector<int> indices(n, -1);
  for (size_t i = 0; i < n; ++i) {
    RefToBase<Track> track(tracks, i);
    RecoToSimCollection::const_iterator f = associations.find(track);
    if (f != associations.end()) {
      TrackingParticleRef tp = f->val.front().first;
      TrackingParticle::genp_iterator j, b = tp->genParticle_begin(), e = tp->genParticle_end();
      for (j = b; j != e; ++j) {
        const reco::GenParticle *p = j->get();
        if (p->status() == 1) {
          indices[i] = j->key();
          break;
        }
      }
    }
  }
  filler.insert(tracks, indices.begin(), indices.end());
  filler.fill();
  evt.put(std::move(match));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MCTrackMatcher);
