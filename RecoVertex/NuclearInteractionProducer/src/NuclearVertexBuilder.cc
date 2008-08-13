#include "RecoVertex/NuclearInteractionProducer/interface/NuclearVertexBuilder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

void NuclearVertexBuilder::build( const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks) {

     FilldistanceOfClosestApproach( primTrack, secTracks);

     if( secTracks.size() != 0) {
         if( FillVertexWithAdaptVtxFitter(primTrack, secTracks) ) return;
         else if( FillVertexWithCrossingPoint(primTrack, secTracks) ) return;
         else FillVertexWithLastPrimHit( primTrack, secTracks);
     }
     else {
       // if no secondary tracks : vertex position = position of last rechit of the primary track 
       FillVertexWithLastPrimHit( primTrack, secTracks);
     }
}
          
FreeTrajectoryState NuclearVertexBuilder::getTrajectory(const reco::TrackRef& track) const
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

void NuclearVertexBuilder::FillVertexWithLastPrimHit(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks) {
   LogDebug("NuclearInteractionMaker") << "Vertex build from the end point of the primary track";
       the_vertex = reco::Vertex(reco::Vertex::Point(primTrack->outerPosition().x(),
                                         primTrack->outerPosition().y(),
                                         primTrack->outerPosition().z()),
                                         reco::Vertex::Error(), 0.,0.,0);
       the_vertex.add(reco::TrackBaseRef( primTrack ), 1.0);
       for( unsigned short i=0; i != secTracks.size(); i++) {
             if( isGoodSecondaryTrack(secTracks[i], i) )
                 the_vertex.add(reco::TrackBaseRef( secTracks[i] ), 0.0);
       }
}

bool NuclearVertexBuilder::FillVertexWithAdaptVtxFitter(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks) {
         std::vector<reco::TransientTrack> transientTracks;
         transientTracks.push_back( theTransientTrackBuilder->build(primTrack));
         // get the secondary track with the max number of hits
         for( unsigned short i=0; i != secTracks.size(); i++ ) {
                 if( isGoodSecondaryTrack(secTracks[i], i) )
                    transientTracks.push_back( theTransientTrackBuilder->build( secTracks[i]) );
         }
         if( transientTracks.size() == 1 ) return  false;
         AdaptiveVertexFitter AVF;
         try {
            TransientVertex tv = AVF.vertex(transientTracks);
            the_vertex = reco::Vertex(tv);
         }
         catch(VertexException& exception){
            // AdaptivevertexFitter does not work
            LogDebug("NuclearInteractionMaker") << exception.what() << "\n";
            return false;
         }
         catch( cms::Exception& exception){
            // AdaptivevertexFitter does not work
            LogDebug("NuclearInteractionMaker") << exception.what() << "\n";
            return false;
         }
         if( the_vertex.isValid() ) {
            LogDebug("NuclearInteractionMaker") << "Try to build from AdaptiveVertexFitter with " << the_vertex.tracksSize() << " tracks";
            return true;
         }
         else return false;
}


bool NuclearVertexBuilder::FillVertexWithCrossingPoint(const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks) {
            FreeTrajectoryState primTraj = getTrajectory(primTrack);

            // get the secondary track with the max number of hits
            unsigned short maxHits = 0; int indice=-1;
            for( unsigned short i=0; i < secTracks.size(); i++) {
                   // Add references to daughters
                   unsigned short nhits = secTracks[i]->numberOfValidHits();
                   if( nhits > maxHits && isGoodSecondaryTrack(secTracks[i], i)) { maxHits = nhits; indice=i; }
            }

            // Closest points
            GlobalPoint crossing;
            if( indice!=-1 ) crossing = crossingPoint(indice);
            else return false;

             // Create vertex (creation point)
            the_vertex = reco::Vertex(reco::Vertex::Point(crossing.x(),
                                                          crossing.y(),
                                                          crossing.z()),
                                      reco::Vertex::Error(), 0.,0.,0);

            the_vertex.add(reco::TrackBaseRef( primTrack ), 1.0);
            for( unsigned short i=0; i < secTracks.size(); i++) {
                if( isGoodSecondaryTrack(secTracks[i], i) ){
                  if( i==indice ) the_vertex.add(reco::TrackBaseRef( secTracks[i] ), 1.0);
                  else the_vertex.add(reco::TrackBaseRef( secTracks[i] ), 0.0);
               }
            }
            LogDebug("NuclearInteractionMaker") << "Vertex build from crossing point with track : " << indice;
            return true;
}

void NuclearVertexBuilder::FilldistanceOfClosestApproach( const reco::TrackRef& primTrack, const reco::TrackRefVector& secTracks )  {
            distances_.clear();
            crossingPoints_.clear();
            FreeTrajectoryState primTraj = getTrajectory(primTrack);
            ClosestApproachInRPhi *theApproach = new ClosestApproachInRPhi();
            for( unsigned short i=0; i != secTracks.size(); i++) {
              FreeTrajectoryState secTraj = getTrajectory(secTracks[i]);
              bool status = theApproach->calculate(primTraj,secTraj);
              if( status ) { 
                   distances_.push_back(theApproach->distance());
                   crossingPoints_.push_back(theApproach->crossingPoint());
              }
              else { 
                   distances_.push_back(1000);
                   crossingPoints_.push_back( GlobalPoint( 0.0,0.0,0.0 ) );
              }
            }
            delete theApproach;
            return;
}

bool NuclearVertexBuilder::isGoodSecondaryTrack( const reco::TrackRef& secTrack, int i ) {
          if( distanceOfClosestApproach(i) < minDistFromPrim_ && 
              secTrack->normalizedChi2() < chi2Cut_ ) return true;
          else return false;
} 
