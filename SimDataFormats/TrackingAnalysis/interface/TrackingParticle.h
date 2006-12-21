#ifndef SimDataFormats_TrackingParticle_h
#define SimDataFormats_TrackingParticle_h
/** Concrete TrackingParticle. 
 *  All track parameters are passed in the constructor and stored internally.
 */

#include "DataFormats/Candidate/interface/Particle.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

namespace HepMC {
  class GenParticle;
}
class TrackingVertex;

class TrackingParticle : public reco::Particle {
public:
  /// reference to HepMC::GenParticle
  typedef edm::RefVector<edm::HepMCProduct, HepMC::GenParticle > GenParticleRefVector;
  typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle >       GenParticleRef;
  typedef GenParticleRefVector::iterator		         genp_iterator;
  typedef       std::vector<SimTrack>::const_iterator             g4t_iterator;
//  typedef TrackPSimHitRefToBaseVector::const_iterator            pSH_iterator;

  typedef std::vector<TrackingVertex>                TrackingVertexCollection;
  typedef edm::Ref<TrackingVertexCollection>         TrackingVertexRef;
  typedef edm::RefVector<TrackingVertexCollection>   TrackingVertexRefVector;
  typedef TrackingVertexRefVector::iterator   tv_iterator;

//  typedef TrackPSimHitRefVector::iterator 			 pSH_iterator;
//  typedef std::map<int, TrackPSimHitRefVector> 			 TrackIdPSimHitMap;
  
  /// default constructor
  TrackingParticle() { }
  // destructor
  ~TrackingParticle();
  /// constructor from pointer to generator particle
  TrackingParticle( char q, const LorentzVector & p4, const Point & vtx,
		    double t, const int pdgId,  const EncodedEventId eventId);
  
  /// PDG id, signal source, crossing number  
  int pdgId() const { return pdgId_; }
  EncodedEventId eventId() const { return eventId_; }
  
  ///iterators
  genp_iterator genParticle_begin() const;
  genp_iterator genParticle_end()   const;
  g4t_iterator  g4Track_begin()     const;
  g4t_iterator  g4Track_end()       const;
  
  const std::vector<PSimHit>::const_iterator  pSimHit_begin() const;
  const std::vector<PSimHit>::const_iterator  pSimHit_end()   const;

// Setters for G4 and HepMC
  void addG4Track(    const SimTrack&);
  void addGenParticle(const GenParticleRef&);

  void addPSimHit(const PSimHit&);
  void setParentVertex(const TrackingVertexRef&);
  void  addDecayVertex(const TrackingVertexRef&);
  void setMatchedHit(const int&);
  void setVertex(const Point & vtx, double t);
  
// Getters for Embd and Sim Tracks
  GenParticleRefVector	 genParticle() const { return genParticles_; }
  std::vector<SimTrack>     g4Tracks() const { return g4Tracks_ ;    }
  std::vector<PSimHit> trackPSimHit() const { return trackPSimHit_; }
  TrackingVertexRef parentVertex() const { return parentVertex_; }

// Accessors for vector of decay vertices    
//  TrackingVertexRefVector decayVertices() const { return decayVertices_; }
  tv_iterator decayVertices_begin()       const { return decayVertices_.begin(); }
  tv_iterator decayVertices_end()         const { return decayVertices_.end(); }

  int matchedHit() const {return matchedHit_;}

private:
  /// production time
  double t_;
  /// PDG identifier, signal source, crossing number
  int pdgId_;
  EncodedEventId eventId_;

  /// Number of Hit to be used for tracking Study
  int matchedHit_;

  /// references to G4 and HepMC tracks
  std::vector<SimTrack> g4Tracks_;
  GenParticleRefVector  genParticles_;
 
//  TrackPSimHitRefVector trackPSimHit_;
  std::vector<PSimHit> trackPSimHit_;
  
// Source and decay vertices  
  TrackingVertexRef       parentVertex_;
  TrackingVertexRefVector  decayVertices_;
};

#endif // SimDataFormats_TrackingParticle_H
