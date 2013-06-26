#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

const unsigned int TrackingParticle::longLivedTag = 65536;

TrackingParticle::TrackingParticle()
{
	// No operation
}

TrackingParticle::TrackingParticle( const SimTrack& simtrk, const TrackingVertexRef& parentVertex )
{
	addG4Track( simtrk );
	setParentVertex( parentVertex );
}

TrackingParticle::~TrackingParticle()
{
}

int TrackingParticle::pdgId() const
{
	if( genParticles_.empty() ) return g4Tracks_.at( 0 ).type();
	else return (*genParticles_.begin())->pdgId();
}

EncodedEventId TrackingParticle::eventId() const
{
	return g4Tracks_.at( 0 ).eventId();
}

void TrackingParticle::addGenParticle( const reco::GenParticleRef& ref )
{
	genParticles_.push_back( ref );
}

void TrackingParticle::addG4Track( const SimTrack& t )
{
	g4Tracks_.push_back( t );
}

TrackingParticle::genp_iterator TrackingParticle::genParticle_begin() const
{
	return genParticles_.begin();
}

TrackingParticle::genp_iterator TrackingParticle::genParticle_end() const
{
	return genParticles_.end();
}

TrackingParticle::g4t_iterator TrackingParticle::g4Track_begin() const
{
	return g4Tracks_.begin();
}

TrackingParticle::g4t_iterator TrackingParticle::g4Track_end() const
{
	return g4Tracks_.end();
}

void TrackingParticle::setParentVertex( const TrackingVertexRef& ref )
{
	parentVertex_=ref;
}

void TrackingParticle::addDecayVertex( const TrackingVertexRef& ref )
{
	decayVertices_.push_back( ref );
}

void TrackingParticle::clearParentVertex()
{
	parentVertex_=TrackingVertexRef();
}

void TrackingParticle::clearDecayVertices()
{
	decayVertices_.clear();
}

const reco::GenParticleRefVector& TrackingParticle::genParticles() const
{
	return genParticles_;
}

const std::vector<SimTrack>& TrackingParticle::g4Tracks() const
{
	return g4Tracks_;
}

const TrackingVertexRef& TrackingParticle::parentVertex() const
{
	return parentVertex_;
}

const TrackingVertexRefVector& TrackingParticle::decayVertices() const
{
	return decayVertices_;
}

tv_iterator TrackingParticle::decayVertices_begin() const
{
	return decayVertices_.begin();
}

tv_iterator TrackingParticle::decayVertices_end() const
{
	return decayVertices_.end();
}

int TrackingParticle::charge() const
{
	return g4Tracks_.at( 0 ).charge();
}

int TrackingParticle::threeCharge() const
{
	return g4Tracks_.at( 0 ).charge()*3;
}

const TrackingParticle::LorentzVector& TrackingParticle::p4() const
{
	return g4Tracks_.at( 0 ).momentum();
}

TrackingParticle::Vector TrackingParticle::momentum() const
{
	return p4().Vect();
}

TrackingParticle::Vector TrackingParticle::boostToCM() const
{
	return p4().BoostToCM();
}

double TrackingParticle::p() const
{
	return p4().P();
}

double TrackingParticle::energy() const
{
	return p4().E();
}

double TrackingParticle::et() const
{
	return p4().Et();
}

double TrackingParticle::mass() const
{
	return p4().M();
}

double TrackingParticle::massSqr() const
{
	return pow( mass(), 2 );
}

double TrackingParticle::mt() const
{
	return p4().Mt();
}

double TrackingParticle::mtSqr() const
{
	return p4().Mt2();
}

double TrackingParticle::px() const
{
	return p4().Px();
}

double TrackingParticle::py() const
{
	return p4().Py();
}

double TrackingParticle::pz() const
{
	return p4().Pz();
}

double TrackingParticle::pt() const
{
	return p4().Pt();
}

double TrackingParticle::phi() const
{
	return p4().Phi();
}

double TrackingParticle::theta() const
{
	return p4().Theta();
}

double TrackingParticle::eta() const
{
	return p4().Eta();
}

double TrackingParticle::rapidity() const
{
	return p4().Rapidity();
}

double TrackingParticle::y() const
{
	return rapidity();
}

TrackingParticle::Point TrackingParticle::vertex() const
{
	return Point( vx(), vy(), vz() );
}

double TrackingParticle::vx() const
{
	const TrackingVertex& r=( *parentVertex_);
	return r.position().X();
}

double TrackingParticle::vy() const
{
	const TrackingVertex& r=( *parentVertex_);
	return r.position().Y();
}

double TrackingParticle::vz() const
{
	const TrackingVertex& r=( *parentVertex_);
	return r.position().Z();
}

int TrackingParticle::status() const
{
	if( genParticles_.empty() ) return -99; // Use the old invalid status flag that used to be set by TrackingTruthProducer.
	else return (*genParticles_.begin())->status();
}

bool TrackingParticle::longLived() const
{
	return status()&longLivedTag;
}

int TrackingParticle::numberOfHits() const
{
    return numberOfHits_;
}

int TrackingParticle::numberOfTrackerHits() const
{
    return numberOfTrackerHits_;
}

int TrackingParticle::matchedHit() const
{
	edm::LogWarning("TrackingParticle") << "The method matchedHit() has been deprecated. Use numberOfTrackerLayers() instead.";
	return numberOfTrackerLayers_;
}

int TrackingParticle::numberOfTrackerLayers() const
{
	return numberOfTrackerLayers_;
}

void TrackingParticle::setNumberOfHits( int numberOfHits )
{
    numberOfHits_=numberOfHits;
}

void TrackingParticle::setNumberOfTrackerHits( int numberOfTrackerHits )
{
    numberOfTrackerHits_=numberOfTrackerHits;
}

void TrackingParticle::setNumberOfTrackerLayers( const int numberOfTrackerLayers )
{
	numberOfTrackerLayers_=numberOfTrackerLayers;
}

std::ostream& operator<< (std::ostream& s, TrackingParticle const & tp)
{
    s << "TP momentum, q, ID, & Event #: "
    << tp.p4()                      << " " << tp.charge() << " "   << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;

    for (TrackingParticle::genp_iterator hepT = tp.genParticle_begin(); hepT !=  tp.genParticle_end(); ++hepT)
    {
        s << " HepMC Track Momentum " << (*hepT)->momentum().rho() << std::endl;
    }

    for (TrackingParticle::g4t_iterator g4T = tp.g4Track_begin(); g4T !=  tp.g4Track_end(); ++g4T)
    {
        s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
        s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
        if (g4T->type() !=  tp.pdgId())
        {
            s << " Mismatch b/t TrackingParticle and Geant types" << std::endl;
        }
    }
    // Loop over decay vertices
    s << " TP Vertex " << tp.vertex() << std::endl;
    s << " Source vertex: " << tp.parentVertex()->position() << std::endl;
    s << " " << tp.decayVertices().size() << " Decay vertices" << std::endl;
    for (tv_iterator iTV = tp.decayVertices_begin(); iTV != tp.decayVertices_end(); ++iTV)
    {
        s << " Decay vertices:      " << (**iTV).position() << std::endl;
    }

    return s;
}
