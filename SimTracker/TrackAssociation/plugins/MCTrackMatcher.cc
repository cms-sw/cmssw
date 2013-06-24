/** \class MCTrackMatcher 
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: MCTrackMatcher.cc,v 1.7.2.1 2013/05/14 14:47:44 speer Exp $
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
  void produce( edm::Event& evt, const edm::EventSetup& es );
  std::string associator_;
  edm::InputTag tracks_, genParticles_, trackingParticles_;
  typedef edm::Association<reco::GenParticleCollection> GenParticleMatch;
};

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
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
}

void MCTrackMatcher::produce(Event& evt, const EventSetup& es) {
  ESHandle<TrackAssociatorBase> assoc;  
  es.get<TrackAssociatorRecord>().get(associator_,assoc);
  const TrackAssociatorBase * associator = assoc.product();
  Handle<View<Track> > tracks;
  evt.getByLabel(tracks_, tracks);
  Handle<TrackingParticleCollection> trackingParticles;
  evt.getByLabel(trackingParticles_,trackingParticles);
  Handle<vector<int> > barCodes;
  evt.getByLabel(genParticles_,barCodes );
  Handle<GenParticleCollection> genParticles;
  evt.getByLabel(genParticles_, genParticles );
  RecoToSimCollection associations = associator->associateRecoToSim ( tracks, trackingParticles, & evt, &es ); 
  auto_ptr<GenParticleMatch> match(new GenParticleMatch(GenParticleRefProd(genParticles)));
  GenParticleMatch::Filler filler(*match);
  size_t n = tracks->size();
  vector<int> indices(n,-1);
  for (size_t i = 0; i < n; ++ i ) {
    RefToBase<Track> track(tracks, i);
    RecoToSimCollection::const_iterator f = associations.find(track);
    if ( f != associations.end() ) {
      TrackingParticleRef tp = f->val.front().first;
      const HepMC::GenParticle * particle = 0;
      TrackingParticle::genp_iterator j, b = tp->genParticle_begin(), e = tp->genParticle_end();
      for( j = b; j != e; ++ j ) {
#warning "This file has been modified just to get it to compile without any regard as to whether it still functions as intended"
#ifdef REMOVED_JUST_TO_GET_IT_TO_COMPILE__THIS_CODE_NEEDS_TO_BE_CHECKED
	const HepMC::GenParticle * p = j->get();
	if (p->status() == 1) {
	  particle = p; break;
	}
#endif
      }
      if( particle != 0 ) {
	int barCode = particle->barcode();
	vector<int>::const_iterator 
	  b = barCodes->begin(), e = barCodes->end(), f = find( b, e, barCode );
	if(f == e) throw edm::Exception(errors::InvalidReference)
	  << "found matching particle with barcode" << *f
	  << " which has not been found in " << genParticles_;
	indices[i] = *f;
      }
    }
  }
  filler.insert(tracks, indices.begin(), indices.end());
  filler.fill();
  evt.put(match);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTrackMatcher );

