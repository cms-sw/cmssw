#ifndef NuclearVertexBuilder_h_
#define NuclearVertexBuilder_h_

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FreeTrajectoryState;

class NuclearVertexBuilder {

  public :
       NuclearVertexBuilder( const MagneticField * mag, const TransientTrackBuilder* transientTkBuilder, const edm::ParameterSet& iConfig ) : 
               theMagField(mag), 
               theTransientTrackBuilder(transientTkBuilder), 
               minDistFromPrim_( iConfig.getParameter<double>("minDistFromPrimary") ),
               chi2Cut_(iConfig.getParameter<double>("chi2Cut")) {}

       void build( const reco::TrackRef& primaryTrack, const reco::TrackRefVector& secondaryTrack );
       reco::Vertex  getVertex() const { return the_vertex; } 

       double distanceOfClosestApproach(int i) const { return distances_[i]; }
       GlobalPoint crossingPoint(int i) const { return crossingPoints_[i]; }

       bool isGoodSecondaryTrack( const reco::TrackRef& secTrack, int i );

  private :
       FreeTrajectoryState getTrajectory(const reco::TrackRef& track) const;
       bool FillVertexWithCrossingPoint(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks);
       bool FillVertexWithAdaptVtxFitter(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks);
       void FillVertexWithLastPrimHit(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks);
       void FilldistanceOfClosestApproach( const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks );

       reco::Vertex  the_vertex;

       const MagneticField * theMagField;
       const TransientTrackBuilder* theTransientTrackBuilder;
       double minDistFromPrim_;
       double chi2Cut_;

       std::vector<double> distances_;  // distance of closest approach
       std::vector<GlobalPoint> crossingPoints_; // crossing points
};

#endif
