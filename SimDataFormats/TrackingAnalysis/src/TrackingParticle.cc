#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

typedef edm::Ref<TrackingVertexCollection>         TrackingVertexRef;

TrackingParticle::TrackingParticle( float q, const LorentzVector & p4, const Point & vtx,
				    double t, const int pdgId, const EncodedEventId eventId) :
  reco::Particle( q, p4, vtx ), t_( t ), pdgId_( pdgId ), eventId_( eventId ){
}

TrackingParticle::~TrackingParticle() { 
}

void TrackingParticle::addGenParticle( const edm::Ref<edm::HepMCProduct, HepMC::GenParticle > &ref) { 
  genParticles_.push_back(ref);
}

void TrackingParticle::addG4Track( const SimTrack& t) { 
  g4Tracks_.push_back(t);
}

//void TrackingParticle::addPSimHit( const TrackPSimHitRef& ref){
//  trackPSimHit_.push_back(ref);
//}

void TrackingParticle::addPSimHit( const PSimHit& hit){
  trackPSimHit_.push_back(hit);
}

TrackingParticle::genp_iterator TrackingParticle::genParticle_begin() const {
   return genParticles_.begin();
}

TrackingParticle::genp_iterator TrackingParticle::genParticle_end() const {
   return genParticles_.end();
}

TrackingParticle::g4t_iterator TrackingParticle::g4Track_begin() const {
    return g4Tracks_.begin();
}

TrackingParticle::g4t_iterator TrackingParticle::g4Track_end() const {
    return g4Tracks_.end();
}

const std::vector<PSimHit>::const_iterator TrackingParticle::pSimHit_begin() const {
    return trackPSimHit_.begin();
}

const std::vector<PSimHit>::const_iterator TrackingParticle::pSimHit_end() const {
    return trackPSimHit_.end();
}

void TrackingParticle::setParentVertex(const TrackingVertexRef &ref) {
  parentVertex_ = ref;
}  

void TrackingParticle::setDecayVertex(const TrackingVertexRef &ref) {
  decayVertex_ = ref;
}  

void TrackingParticle::setMatchedHit(const int &hitnumb) {
  matchedHit_ = hitnumb;
}
