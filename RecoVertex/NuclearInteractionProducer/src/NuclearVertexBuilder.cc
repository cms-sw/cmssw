#include "RecoVertex/NuclearInteractionProducer/interface/NuclearVertexBuilder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

NuclearVertexBuilder::NuclearVertexBuilder( const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks, const MagneticField * field) : theMagField(field) {


       if( secTracks.size() != 0) {

         FreeTrajectoryState primTraj = getTrajectory(primTrack);

         // get the secondary track with the max number of hits
         unsigned short maxHits = 0; int indice=0;
         for( unsigned short i=0; i < secTracks.size(); i++) { 
                   // Add references to daughters
                   unsigned short nhits = secTracks[i]->numberOfValidHits();
                   if( nhits > maxHits ) { maxHits = nhits; indice=i; }
         }
         FreeTrajectoryState secTraj = getTrajectory(secTracks[indice]);

         // Closest points
         ClosestApproachInRPhi *theApproach = new ClosestApproachInRPhi();
         GlobalPoint crossing = theApproach->crossingPoint(primTraj,secTraj);
         delete theApproach;

         // Create vertex (creation point)
         the_vertex = reco::Vertex(reco::Vertex::Point(crossing.x(),
                                                       crossing.y(),
                                                       crossing.z()),
                                   reco::Vertex::Error(), 0.,0.,0);
 
     }

     // Add all the tracks to the vertex
     the_vertex.add(reco::TrackBaseRef( primTrack ));
     for( unsigned short i=0; i < secTracks.size(); i++) { 
                   the_vertex.add(reco::TrackBaseRef( secTracks[i] ));
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

