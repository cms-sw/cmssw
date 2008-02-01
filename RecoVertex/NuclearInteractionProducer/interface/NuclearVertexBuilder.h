#ifndef NuclearVertexBuilder_h_
#define NuclearVertexBuilder_h_

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class FreeTrajectoryState;

class NuclearVertexBuilder {

  public :
       NuclearVertexBuilder( const MagneticField * mag, const TransientTrackBuilder* transientTkBuilder, float dist ) : theMagField(mag), theTransientTrackBuilder(transientTkBuilder), minDistFromPrim_(dist) {}
       void build( const reco::TrackRef& primaryTrack, const reco::TrackRefVector& secondaryTrack );
       reco::Vertex  getVertex() const { return the_vertex; } 

  private :
       FreeTrajectoryState getTrajectory(const reco::TrackRef& track) const;
       bool FillVertexWithCrossingPoint(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks);
       bool FillVertexWithAdaptVtxFitter(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks);
       void FillVertexWithLastPrimHit(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks);
       float distanceOfClosestApproach( const reco::TrackRef& primTrack, const reco::TrackRef& secTrack ) const ;

       reco::Vertex  the_vertex;

       const MagneticField * theMagField;
       const TransientTrackBuilder* theTransientTrackBuilder;
       float minDistFromPrim_;
};

#endif
