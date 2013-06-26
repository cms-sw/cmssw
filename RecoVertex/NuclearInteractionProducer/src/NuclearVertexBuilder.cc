#include "RecoVertex/NuclearInteractionProducer/interface/NuclearVertexBuilder.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

void NuclearVertexBuilder::build( const reco::TrackRef& primTrack, std::vector<reco::TrackRef>& secTracks) {

     cleanTrackCollection(primTrack, secTracks); 
     std::sort(secTracks.begin(),secTracks.end(),cmpTracks());
     checkEnergy(primTrack, secTracks);

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

void NuclearVertexBuilder::FillVertexWithLastPrimHit(const reco::TrackRef& primTrack, const std::vector<reco::TrackRef>& secTracks) {
   LogDebug("NuclearInteractionMaker") << "Vertex build from the end point of the primary track";
       the_vertex = reco::Vertex(reco::Vertex::Point(primTrack->outerPosition().x(),
                                         primTrack->outerPosition().y(),
                                         primTrack->outerPosition().z()),
                                         reco::Vertex::Error(), 0.,0.,0);
       the_vertex.add(reco::TrackBaseRef( primTrack ), 1.0);
       for( unsigned short i=0; i != secTracks.size(); i++) {
             the_vertex.add(reco::TrackBaseRef( secTracks[i] ), 0.0);
       }
}

bool NuclearVertexBuilder::FillVertexWithAdaptVtxFitter(const reco::TrackRef& primTrack, const std::vector<reco::TrackRef>& secTracks) {
         std::vector<reco::TransientTrack> transientTracks;
         transientTracks.push_back( theTransientTrackBuilder->build(primTrack));
         // get the secondary track with the max number of hits
         for( unsigned short i=0; i != secTracks.size(); i++ ) {
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


bool NuclearVertexBuilder::FillVertexWithCrossingPoint(const reco::TrackRef& primTrack, const std::vector<reco::TrackRef>& secTracks) {
            // get the secondary track with the max number of hits
            unsigned short maxHits = 0; int indice=-1;
            for( unsigned short i=0; i < secTracks.size(); i++) {
                   // Add references to daughters
                   unsigned short nhits = secTracks[i]->numberOfValidHits();
                   if( nhits > maxHits ) { maxHits = nhits; indice=i; }
            }

            // Closest points
            if(indice == -1) return false;

            ClosestApproachInRPhi* theApproach = closestApproach( primTrack, secTracks[indice]);
            GlobalPoint crossing = theApproach->crossingPoint();
            delete theApproach;

            // Create vertex (creation point)
            // TODO: add error
            the_vertex = reco::Vertex(reco::Vertex::Point(crossing.x(),
                                                          crossing.y(),
                                                          crossing.z()),
                                      reco::Vertex::Error(), 0.,0.,0);

            the_vertex.add(reco::TrackBaseRef( primTrack ), 1.0);
            for( unsigned short i=0; i < secTracks.size(); i++) {
                 if( i==indice ) the_vertex.add(reco::TrackBaseRef( secTracks[i] ), 1.0);
                 else the_vertex.add(reco::TrackBaseRef( secTracks[i] ), 0.0);
            }
            return true;
}

ClosestApproachInRPhi* NuclearVertexBuilder::closestApproach( const reco::TrackRef& primTrack, const reco::TrackRef& secTrack )  const{
            FreeTrajectoryState primTraj = getTrajectory(primTrack);
            ClosestApproachInRPhi *theApproach = new ClosestApproachInRPhi();
            FreeTrajectoryState secTraj = getTrajectory(secTrack);
            bool status = theApproach->calculate(primTraj,secTraj);
            if( status ) { return theApproach; }
            else { 
                   return NULL;
            }
}

bool NuclearVertexBuilder::isGoodSecondaryTrack( const reco::TrackRef& primTrack, const reco::TrackRef& secTrack ) const {
           ClosestApproachInRPhi* theApproach = closestApproach(primTrack, secTrack);
           bool result = false;
           if(theApproach)
             result = isGoodSecondaryTrack(secTrack, primTrack, theApproach->distance() , theApproach->crossingPoint());
           delete theApproach;
           return result;
} 

bool NuclearVertexBuilder::isGoodSecondaryTrack( const reco::TrackRef& secTrack, 
                                                 const reco::TrackRef& primTrack, 
                                                 const double& distOfClosestApp, 
                                                 const GlobalPoint& crossPoint ) const {
          float TRACKER_RADIUS=129;
          double pt2 = secTrack->pt();
          double Dpt2 = secTrack->ptError();
          double p1 = primTrack->p();
          double Dp1 = primTrack->qoverpError()*p1*p1;
          double p2 = secTrack->p();
          double Dp2 = secTrack->qoverpError()*p2*p2;
          std::cout << "1)" << distOfClosestApp << " < " << minDistFromPrim_ << " " << (distOfClosestApp < minDistFromPrim_)  << "\n";
          std::cout << "2)" << secTrack->normalizedChi2() << " < " << chi2Cut_ << " " << (secTrack->normalizedChi2() < chi2Cut_) << "\n";
          std::cout << "3)" << crossPoint.perp() << " < " << TRACKER_RADIUS << " " << (crossPoint.perp() < TRACKER_RADIUS) << "\n";
          std::cout << "4)" << (Dpt2/pt2) << " < " << DPtovPtCut_ << " " << ((Dpt2/pt2) < DPtovPtCut_)  << "\n";
          std::cout << "5)" << (p2-2*Dp2) << " < " << (p1+2*Dp1) << " " << ((p2-2*Dp2) < (p1+2*Dp1))<< "\n";
          if( distOfClosestApp < minDistFromPrim_ &&
              secTrack->normalizedChi2() < chi2Cut_ && 
              crossPoint.perp() < TRACKER_RADIUS && 
              (Dpt2/pt2) < DPtovPtCut_ &&
              (p2-2*Dp2) < (p1+2*Dp1)) return true;
          else return false;
}


bool NuclearVertexBuilder::isCompatible( const reco::TrackRef& secTrack ) const{
         
         reco::TrackRef primTrack = (*(the_vertex.tracks_begin())).castTo<reco::TrackRef>();
         ClosestApproachInRPhi *theApproach = closestApproach(primTrack, secTrack);
         bool result = false;
         if( theApproach ) {
           GlobalPoint crp = theApproach->crossingPoint();
           math::XYZPoint vtx = the_vertex.position();
           float dist = sqrt((crp.x()-vtx.x())*(crp.x()-vtx.x()) +
                        (crp.y()-vtx.y())*(crp.y()-vtx.y()) +
                        (crp.z()-vtx.z())*(crp.z()-vtx.z()));

           float distError = sqrt(the_vertex.xError()*the_vertex.xError() +
                                  the_vertex.yError()*the_vertex.yError() +
                                  the_vertex.zError()*the_vertex.zError());
 
           //std::cout << "Distance between Additional track and vertex =" << dist << " +/- " << distError << std::endl;
           // TODO : add condition on distance between last rechit of the primary and the first rec hit of the secondary
 
           result = (isGoodSecondaryTrack(secTrack, primTrack, theApproach->distance(), crp) 
                     && dist-distError<minDistFromVtx_);
         }
         delete theApproach;
         return result;
}

void NuclearVertexBuilder::addSecondaryTrack( const reco::TrackRef& secTrack ) {
        std::vector<reco::TrackRef> allSecondary;
        for( reco::Vertex::trackRef_iterator it=the_vertex.tracks_begin()+1; it != the_vertex.tracks_end(); it++) {
            allSecondary.push_back( (*it).castTo<reco::TrackRef>() );
        } 
        allSecondary.push_back( secTrack );
        build( (*the_vertex.tracks_begin()).castTo<reco::TrackRef>(), allSecondary );
}

void NuclearVertexBuilder::cleanTrackCollection( const reco::TrackRef& primTrack, 
                                                 std::vector<reco::TrackRef>& tC) const {

    // inspired from FinalTrackSelector (S. Wagner) modified by P. Janot
    LogDebug("NuclearInteractionMaker") << "cleanTrackCollection number of input tracks : " << tC.size();
   std::map<std::vector<reco::TrackRef>::const_iterator, std::vector<const TrackingRecHit*> > rh;

    // first remove bad quality tracks and create map
    std::vector<bool> selected(tC.size(), false);
    int i=0;
    for (std::vector<reco::TrackRef>::const_iterator track=tC.begin(); track!=tC.end(); track++){
       if( isGoodSecondaryTrack(primTrack, *track)) { 
            selected[i]=true;
            trackingRecHit_iterator itB = (*track)->recHitsBegin();
            trackingRecHit_iterator itE = (*track)->recHitsEnd();
            for (trackingRecHit_iterator it = itB;  it != itE; ++it) {
               const TrackingRecHit* hit = &(**it);
               rh[track].push_back(hit);
            }
        }
       i++;
    }

    // then remove duplicated tracks
    i=-1;
    for (std::vector<reco::TrackRef>::const_iterator track=tC.begin(); track!=tC.end(); track++){
      i++;
      int j=-1;
      for (std::vector<reco::TrackRef>::const_iterator track2=tC.begin(); track2!=tC.end(); track2++){
        j++;
        if ((!selected[j])||(!selected[i]))continue;
        if ((j<=i))continue;
        int noverlap=0;
        std::vector<const TrackingRecHit*>& iHits = rh[track];
        for ( unsigned ih=0; ih<iHits.size(); ++ih ) {
          const TrackingRecHit* it = iHits[ih];
          if (it->isValid()){
            std::vector<const TrackingRecHit*>& jHits = rh[track2];
            for ( unsigned ih2=0; ih2<jHits.size(); ++ih2 ) {
            const TrackingRecHit* jt = jHits[ih2];
              if (jt->isValid()){
                const TrackingRecHit* kt = jt;
                if ( it->sharesInput(kt,TrackingRecHit::some) )noverlap++;
               }
             }
          }
        }
        float fi=float(noverlap)/float((*track)->recHitsSize()); 
        float fj=float(noverlap)/float((*track2)->recHitsSize());
        if ((fi>shareFrac_)||(fj>shareFrac_)){
          if (fi<fj){
            selected[j]=false;
          }else{
            if (fi>fj){
              selected[i]=false;
            }else{
              if ((*track)->normalizedChi2() > (*track2)->normalizedChi2()){selected[i]=false;}else{selected[j]=false;}
            }//end fi > or = fj
          }//end fi < fj
        }//end got a duplicate
      }//end track2 loop
    }//end track loop

   std::vector< reco::TrackRef > newTrackColl;
   i=0;
   for (std::vector<reco::TrackRef>::const_iterator track=tC.begin(); track!=tC.end(); track++){
         if( selected[i] ) newTrackColl.push_back( *track );
         ++i;
   }
   tC = newTrackColl;
}

void NuclearVertexBuilder::checkEnergy( const reco::TrackRef& primTrack,
                                        std::vector<reco::TrackRef>& tC) const {
   float totalEnergy=0;
   for(size_t i=0; i< tC.size(); ++i) {
     totalEnergy += tC[i]->p();
   }
   if( totalEnergy > primTrack->p()+0.1*primTrack->p() ) {
           tC.pop_back();
           checkEnergy(primTrack,tC);
   }
}
