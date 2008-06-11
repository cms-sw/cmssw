#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

typedef std::vector<TrackingVertex>                TrackingVertexCollection;
typedef edm::Ref<TrackingVertexCollection>         TrackingVertexRef;
typedef edm::RefVector<TrackingVertexCollection>   TrackingVertexRefVector;
typedef TrackingVertexRefVector::iterator          tv_iterator;

TrackingParticle::TrackingParticle( char q, const LorentzVector & p4, const Point & vtx,
                                    double t, const int pdgId, const int status, const EncodedEventId eventId) :
  reco::Particle( q, p4, vtx,pdgId,status ), t_( t ), pdgId_( pdgId ), eventId_( eventId ), subdetVectorFill_( false ) {
}

TrackingParticle::~TrackingParticle() {
}

void TrackingParticle::addGenParticle( const edm::Ref<edm::HepMCProduct, HepMC::GenParticle > &ref) {
  genParticles_.push_back(ref);
}

void TrackingParticle::addG4Track( const SimTrack& t) {
  g4Tracks_.push_back(t);
}

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

const std::vector<PSimHit>::const_iterator TrackingParticle::trackerPSimHit_begin() {
  /* do it because PSimHit vectors
     trackerPSimHit_ and muonPSimHit_
     are transient */
  if(!subdetVectorFill_)
    fillSubDetHitVectors();
  
  return trackerPSimHit_.begin();
}

const std::vector<PSimHit>::const_iterator TrackingParticle::trackerPSimHit_end() {
  /* do it because PSimHit vectors
     trackerPSimHit_ and muonPSimHit_
     are transient */
  if(!subdetVectorFill_)
    fillSubDetHitVectors();
  
  return trackerPSimHit_.end();
}

const std::vector<PSimHit>::const_iterator TrackingParticle::muonPSimHit_begin() {
  /* do it because PSimHit vectors
     trackerPSimHit_ and muonPSimHit_
     are transient */
  if(!subdetVectorFill_)
    fillSubDetHitVectors();
  
  return muonPSimHit_.begin();
}

const std::vector<PSimHit>::const_iterator TrackingParticle::muonPSimHit_end() {
  /* do it because PSimHit vectors
     trackerPSimHit_ and muonPSimHit_
     are transient */
  if(!subdetVectorFill_)
    fillSubDetHitVectors();
  
  return muonPSimHit_.end();
}

const std::vector<PSimHit>& TrackingParticle::trackerPSimHit() {
  /* do it because PSimHit vectors
     trackerPSimHit_ and muonPSimHit_
     are transient */
  if(!subdetVectorFill_)
    fillSubDetHitVectors();
  
  return trackerPSimHit_;
}

const std::vector<PSimHit>& TrackingParticle::muonPSimHit() {
  /* do it because PSimHit vectors
     trackerPSimHit_ and muonPSimHit_
     are transient */
  if(!subdetVectorFill_)
    fillSubDetHitVectors();
  
  return muonPSimHit_;
}

void TrackingParticle::fillSubDetHitVectors() {
  trackerPSimHit_.clear();
  muonPSimHit_.clear();
  subdetVectorFill_ = true;
  //
  for(std::vector<PSimHit>::const_iterator aHit = trackPSimHit_.begin(); aHit != trackPSimHit_.end(); ++aHit) {
    unsigned int subDet_enum = DetId( (uint32_t)((*aHit).detUnitId()) ).det();
    switch (subDet_enum) {
    case DetId::Tracker:
      {
	trackerPSimHit_.push_back((*aHit));
	break;
      }
    case DetId::Muon:
      {
	muonPSimHit_.push_back((*aHit));
	break;
      }
    default: 
      {
      std::cout << "TrackingParticle WARNING - Not Tracker or Muon Subdetector " << subDet_enum << std::endl;
      break;
      }
    } // switch
  }
}


void TrackingParticle::setParentVertex(const TrackingVertexRef &ref) {
  parentVertex_ = ref;
}

void TrackingParticle::addDecayVertex(const TrackingVertexRef &ref){
  decayVertices_.push_back(ref);
}

void TrackingParticle::clearParentVertex() {
  parentVertex_ = TrackingVertexRef();
}

void TrackingParticle::clearDecayVertices() {
  decayVertices_.clear();
}

void TrackingParticle::setMatchedHit(const int &hitnumb) {
  matchedHit_ = hitnumb;
}

void TrackingParticle::setVertex(const Point & vtx, double t){
  t_ = t;
  reco::Particle::setVertex(vtx);
}

std::ostream& operator<< (std::ostream& s, const TrackingParticle & tp) {
  
  // Compare momenta from sources
  s << "TP momentum, q, ID, & Event #: "
    << tp.p4()                      << " " << tp.charge() << " "   << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;
  s << " Hits for this track: " << tp.trackPSimHit().size() << std::endl;
  s << "\t Tracker: " << tp.trackerPSimHit_.size() << std::endl;
  s << "\t Muon: "    << tp.muonPSimHit_.size()    << std::endl;
  
  for (TrackingParticle::genp_iterator hepT = tp.genParticle_begin(); hepT !=  tp.genParticle_end(); ++hepT) {
    s << " HepMC Track Momentum " << (*hepT)->momentum().mag() << std::endl;
  }
  
  for (TrackingParticle::g4t_iterator g4T = tp.g4Track_begin(); g4T !=  tp.g4Track_end(); ++g4T) {
    s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
    s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
    if (g4T->type() !=  tp.pdgId()) {
      s << " Mismatch b/t TrackingParticle and Geant types" << std::endl;
    }
  }
  // Loop over decay vertices
  s << " TP Vertex " << tp.vertex() << std::endl;
  s << " Source vertex: " << tp.parentVertex()->position() << std::endl;
  s << " " << tp.decayVertices().size() << " Decay vertices" << std::endl;
  for (tv_iterator iTV = tp.decayVertices_begin(); iTV != tp.decayVertices_end(); ++iTV) {
    s << " Decay vertices:      " << (**iTV).position() << std::endl;
  }
  
  return s;
}
