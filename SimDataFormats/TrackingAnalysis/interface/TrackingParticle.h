#ifndef SimDataFormats_TrackingParticle_H
#define SimDataFormats_TrackingParticle_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/TrackReco/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"  
#include "SimDataFormats/Track/interface/EmbdSimTrack.h"
#include "SimG4Core/Generators/interface/HepMCParticle.h"

#include <vector>

typedef std::vector<TrackingParticle> TrackingParticleContainer;

typedef edm::Ref< std::vector<TrackingVertex> > TrackingVertexRef;

typedef  edm::RefVector< std::vector<PSimHit> > PSimHitCollection;

typedef edm::Ref< std::vector<HepMCParticle> > HepMCParticleRef; // link in HepMC

/** Concrete TrackingParticle. 
 *  All track parameters are passed in the constructor and stored internally.
 */

class TrackingParticle : public Track {

public:

  TrackingParticle(const PSimHitCollection&,
	     unsigned int number, 
	     TrackingVertexRef originVertex, 
	     TrackingVertexRef endVertex,
	     const Vector & momentum, 
	     const TrackCharge& charge,
	     int type, int bunch, 
	     bool fromSignalEvent,
	     const HepMCParticleRef & particle);

  TrackingParticle(const PSimHitCollection&,
	     unsigned int number, 
	     TrackingVertexRef originVertex, 
	     TrackingVertexRef endVertex,
	     const Vector & momentum,
	     const TrackCharge& charge,
	     int type,	int bunch,
	     bool fromSignalEvent,
	     const HepMCParticleRef & particle, 
	     const TrackingParticleContainer& components);
  
  TrackingParticle(const PSimHitCollection&,
	     unsigned int number, 
	     TrackingVertexRef originVertex, 
	     TrackingVertexRef endVertex,
	     const Vector & momentum, 
	     const TrackCharge& charge,
	     int type, int bunch, 
	     bool fromSignalEvent,
	     const EmbdSimTrack & particle);

  TrackingParticle(const PSimHitCollection&,
	     unsigned int number, 
	     TrackingVertexRef originVertex, 
	     TrackingVertexRef endVertex,
	     const Vector & momentum,
	     const TrackCharge& charge,
	     int type,	int bunch,
	     bool fromSignalEvent,
	     const EmbdSimTrack & particle, 
	     const TrackingParticleContainer& components);
  
  unsigned int number() const { return number_;}

  TrackCharge charge() const { return charge_;}
  int type() const { return type_;}
  int bunch() const { return bunch_;}

  bool fromSignalEvent() const {return fromSignalEvent_; }

  Parameters helixParameters() const;

  // access to HepMC particle
  const HepMCParticleRef genParticle() const {return genParticle_;}

  // for particles created by Geant, that have no HepMC link
  const EmbdSimTrack * geantParticle() const {return geantParticle_;}

  TrackingVertexRef originVertex() const {return originVertex_;}
  TrackingVertexRef endVertex() const {return endVertex_;}
  Vector momentumAtOrigin() const { return momentum_;}
  TrackingParticleContainer tracks() const {return componentTracks_;}

  PSimHitCollection hits() const {return hitCollection_;}

  // methods to ease comparison with reconstructed parameters
  //  TrajectoryStateOnSurface outermostMeasurementState() const;
  //  TrajectoryStateOnSurface innermostMeasurementState() const;
  //  TrajectoryStateClosestToPoint 
  //    trajectoryStateClosestToPoint( const GlobalPoint & point ) const;
  //  TrajectoryStateOnSurface impactPointState() const;


private:

  unsigned int           number_;
  Vector                 momentum_;
  TrackCharge            charge_;
  int                    type_;
  int                    bunch_;
  bool fromSignalEvent_;
  TrackingVertexRef originVertex_;       // origin vertex of track
  TrackingVertexRef endVertex_;         // end vertex of track
  HepMCParticleRef genParticle_;    // link to particle in generator tree
  const EmbdSimTrack * geantParticle_;      // in case there is no generator particle, store geant particle
  PSimHitCollection hitCollection_;  // links to hits
  const TrackingParticleContainer& componentTracks_; // TrackingParticle may be composite: whenever GEANT creates several segments for a single particle (radiating electrons etc.), these are merged. The componentTracks vector contains all GEANT segments  
 
};

#endif // SimDataFormats_TrackingParticle_H
