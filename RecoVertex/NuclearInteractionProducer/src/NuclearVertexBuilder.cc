#include "RecoVertex/NuclearInteractionProducer/interface/NuclearVertexBuilder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

void NuclearVertexBuilder::build( const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks) {

       if( secTracks.size() != 0) {
         std::vector<reco::TransientTrack> transientTracks;
         transientTracks.push_back( theTransientTrackBuilder->build(&(*primTrack)));
         // get the secondary track with the max number of hits
         for( reco::TrackRefVector::const_iterator tk = secTracks.begin(); tk != secTracks.end(); tk++) { 
                    transientTracks.push_back( theTransientTrackBuilder->build(*tk));
         }
         AdaptiveVertexFitter AVF;
         TransientVertex tv = AVF.vertex(transientTracks); 
         the_vertex = reco::Vertex(tv);
     }
     else {
       // if no secondary tracks : vertex position = position of last rechit of the primary track 
       the_vertex = reco::Vertex(reco::Vertex::Point(primTrack->outerPosition().x(),
                                         primTrack->outerPosition().y(),
                                         primTrack->outerPosition().z()),
                                         reco::Vertex::Error(), 0.,0.,0);
       the_vertex.add(reco::TrackBaseRef( primTrack ));
     }
}
          
FreeTrajectoryState NuclearVertexBuilder::getTrajectory(const reco::TrackRef& track)
{ 
  GlobalPoint position(track->vertex().x(),
                       track->vertex().y(),
                       track->vertex().z());

  GlobalVector momentum(track->momentum().x(),
                        track->momentum().y(),
                        track->momentum().z());

  GlobalTrajectoryParameters gtp(position,momentum,
                                 track->charge(),theMagField);
  
  FreeTrajectoryState fts(gtp);

  return fts; 
} 

