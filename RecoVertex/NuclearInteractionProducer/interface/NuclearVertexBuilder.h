#ifndef NuclearVertexBuilder_h_
#define NuclearVertexBuilder_h_

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FreeTrajectoryState;

class NuclearVertexBuilder {

  public :
       NuclearVertexBuilder( const MagneticField * mag, const TransientTrackBuilder* transientTkBuilder, const edm::ParameterSet& iConfig ) : 
               theMagField(mag), 
               theTransientTrackBuilder(transientTkBuilder), 
               minDistFromPrim_( iConfig.getParameter<double>("minDistFromPrimary") ),
               chi2Cut_(iConfig.getParameter<double>("chi2Cut")), 
               DPtovPtCut_(iConfig.getParameter<double>("DPtovPtCut")),
               minDistFromVtx_(iConfig.getParameter<double>("minDistFromVtx")),
               shareFrac_(iConfig.getParameter<double>("shareFrac")){}

       void build( const reco::TrackRef& primaryTrack, std::vector<reco::TrackRef>& secondaryTrack );
       reco::Vertex  getVertex() const { return the_vertex; } 
       bool isCompatible( const reco::TrackRef& secTrack ) const;
       void addSecondaryTrack( const reco::TrackRef& secTrack );
       ClosestApproachInRPhi* closestApproach( const reco::TrackRef& primTrack, const reco::TrackRef& secTrack ) const;


  private :
       FreeTrajectoryState getTrajectory(const reco::TrackRef& track) const;
       bool FillVertexWithCrossingPoint(const reco::TrackRef& primTrack, const std::vector<reco::TrackRef>& secTracks);
       bool FillVertexWithAdaptVtxFitter(const reco::TrackRef& primTrack, const std::vector<reco::TrackRef>& secTracks);
       void FillVertexWithLastPrimHit(const reco::TrackRef& primTrack, const std::vector<reco::TrackRef>& secTracks);
       bool isGoodSecondaryTrack( const reco::TrackRef& primTrack,  const reco::TrackRef& secTrack ) const;
       bool isGoodSecondaryTrack( const reco::TrackRef& secTrack,
                                  const reco::TrackRef& primTrack,
                                  const double& distOfClosestApp,
                                  const GlobalPoint& crossPoint ) const;
       void cleanTrackCollection( const reco::TrackRef& primTrack,
                                  std::vector<reco::TrackRef>& tC) const;
       void checkEnergy( const reco::TrackRef& primTrack,
                         std::vector<reco::TrackRef>& tC) const;

       reco::Vertex  the_vertex;


       const MagneticField * theMagField;
       const TransientTrackBuilder* theTransientTrackBuilder;
       double minDistFromPrim_;
       double chi2Cut_;
       double DPtovPtCut_;
       double minDistFromVtx_;
       double shareFrac_;

       class cmpTracks {
          public:
                bool operator () (const reco::TrackRef& a, const reco::TrackRef& b) {
                  if( a->numberOfValidHits() != b->numberOfValidHits()) return (a->numberOfValidHits()> b->numberOfValidHits());
                  else return (a->normalizedChi2()<b->normalizedChi2());
                }
        };

};

#endif
