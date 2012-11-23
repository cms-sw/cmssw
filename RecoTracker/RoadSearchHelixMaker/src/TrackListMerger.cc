//
// Package:         RecoTracker/TrackListMerger
// Class:           RoadSearchHelixMaker
// 
// Description:     TrackList Cleaner and Merger
//
// Original Author: Steve Wagner, stevew@pizero.colorado.edu
// Created:         Sat Jan 14 22:00:00 UTC 2006
//
// $Author: wdd $
// $Date: 2011/04/01 21:52:16 $
// $Revision: 1.4 $
//

#include <memory>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>

#include "RecoTracker/RoadSearchHelixMaker/interface/TrackListMerger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

namespace cms
{

  TrackListMerger::TrackListMerger(edm::ParameterSet const& conf) : 
    conf_(conf)
  {
    produces<reco::TrackCollection>();
//    produces<reco::TrackExtraCollection>();
  }


  // Virtual destructor needed.
  TrackListMerger::~TrackListMerger() { }  

  // Functions that gets called by framework every event
  void TrackListMerger::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input TrackCollection(s)
    std::string trackProducer1 = conf_.getParameter<std::string>("TrackProducer1");
    std::string trackProducer2 = conf_.getParameter<std::string>("TrackProducer2");

    double maxNormalizedChisq =  conf_.getParameter<double>("MaxNormalizedChisq");
    double minPT =  conf_.getParameter<double>("MinPT");
    unsigned int minFound = (unsigned int)conf_.getParameter<int>("MinFound");
    double epsilon =  conf_.getParameter<double>("Epsilon");
    double shareFrac =  conf_.getParameter<double>("ShareFrac");
  
    //
    // extract tracker geometry
    //
    edm::ESHandle<TrackerGeometry> theG;
    es.get<TrackerDigiGeometryRecord>().get(theG);

//    using namespace reco;

    // get Inputs 
    // if 1 input list doesn't exist, make an empty list, issue a warning, and continue
    // this allows TrackListMerger to be used as a cleaner only if handed just one list
    // if both input lists don't exist, will issue 2 warnings and generate an empty output collection
    //
    const reco::TrackCollection *TC1 = 0;
    try {
      edm::Handle<reco::TrackCollection> trackCollection1;
      e.getByLabel(trackProducer1, trackCollection1);
      TC1 = trackCollection1.product();
      //std::cout << "1st collection " << trackProducer1 << " has "<< TC1->size() << " tracks" << std::endl ;
    }
    catch (edm::Exception const& x) {
      if ( x.categoryCode() == edm::errors::ProductNotFound ) {
          static const reco::TrackCollection s_empty;
          TC1 = &s_empty;
          edm::LogWarning("TrackListMerger") << "1st TrackCollection " << trackProducer1 << " not found; will only clean 2nd TrackCollection " << trackProducer2 ;
      }
    }
    const reco::TrackCollection tC1 = *TC1;

    const reco::TrackCollection *TC2 = 0;
    try {
      edm::Handle<reco::TrackCollection> trackCollection2;
      e.getByLabel(trackProducer2, trackCollection2);
      TC2 = trackCollection2.product();
      //std::cout << "2nd collection " << trackProducer2 << " has "<< TC2->size() << " tracks" << std::endl ;
    }
    catch (edm::Exception const& x) {
      if ( x.categoryCode() == edm::errors::ProductNotFound ) {
          static const reco::TrackCollection s_empty;
          TC2 = &s_empty;
          edm::LogWarning("TrackListMerger") << "2nd TrackCollection " << trackProducer2 << " not found; will only clean 1st TrackCollection " << trackProducer1 ;
      }
    }
    const reco::TrackCollection tC2 = *TC2;

    // Step B: create empty output collection
    std::auto_ptr<reco::TrackCollection> output(new reco::TrackCollection);

  //
  //  no input tracks
  //

//    if ( tC1.empty() ){
//      LogDebug("RoadSearch") << "Found " << output.size() << " clouds.";
//      e.put(output);
//      return;  
//    }

  //
  //  quality cuts first
  // 
    int i;

    std::vector<int> selected1; for (unsigned int i=0; i<tC1.size(); ++i){selected1.push_back(1);}

   if ( 0<tC1.size() ){
      i=-1;
      for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); track++){
        i++;
        if ((short unsigned)track->ndof() < 1){
          selected1[i]=0; 
          //std::cout << "L1Track "<< i << " rejected in TrackListMerger; ndof() < 1" << std::endl ;
          continue;
        }
        if (track->normalizedChi2() > maxNormalizedChisq){
          selected1[i]=0; 
          //std::cout << "L1Track "<< i << " rejected in TrackListMerger; normalizedChi2() > maxNormalizedChisq " << track->normalizedChi2() << " " << maxNormalizedChisq << std::endl ;
          continue;
        }
        if (track->found() < minFound){
          selected1[i]=0; 
          //std::cout << "L1Track "<< i << " rejected in TrackListMerger; found() < minFound " << track->found() << " " << minFound << std::endl ;
          continue;
        }
        if (track->pt() < minPT){
          selected1[i]=0; 
          //std::cout << "L1Track "<< i << " rejected in TrackListMerger; pt() < minPT " << track->pt() << " " << minPT << std::endl ;
          continue;
        }
      }//end loop over tracks
   }//end more than 0 track


    std::vector<int> selected2; for (unsigned int i=0; i<tC2.size(); ++i){selected2.push_back(1);}

   if ( 0<tC2.size() ){
      i=-1;
      for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); track++){
        i++;
        if ((short unsigned)track->ndof() < 1){
          selected2[i]=0; 
          //std::cout << "L2Track "<< i << " rejected in TrackListMerger; ndof() < 1" << std::endl ;
          continue;
        }
        if (track->normalizedChi2() > maxNormalizedChisq){
          selected2[i]=0; 
          //std::cout << "L2Track "<< i << " rejected in TrackListMerger; normalizedChi2() > maxNormalizedChisq " << track->normalizedChi2() << " " << maxNormalizedChisq << std::endl ;
          continue;
        }
        if (track->found() < minFound){
          selected2[i]=0; 
          //std::cout << "L2Track "<< i << " rejected in TrackListMerger; found() < minFound " << track->found() << " " << minFound << std::endl ;
          continue;
        }
        if (track->pt() < minPT){
          selected2[i]=0; 
          //std::cout << "L2Track "<< i << " rejected in TrackListMerger; pt() < minPT " << track->pt() << " " << minPT << std::endl ;
          continue;
        }
      }//end loop over tracks
   }//end more than 0 track

  //
  //  L1 has > 1 track - try merging
  //
   if ( 1<tC1.size() ){
    i=-1;
    for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); track++){
      i++; 
      //std::cout << "Track number "<< i << std::endl ; 
      if (!selected1[i])continue;
      int j=-1;
      for (reco::TrackCollection::const_iterator track2=tC1.begin(); track2!=tC1.end(); track2++){
        j++;
        if ((j<=i)||(!selected1[j])||(!selected1[i]))continue;
        int noverlap=0;
        for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
          if ((*it)->isValid()){
            for (trackingRecHit_iterator jt = track2->recHitsBegin();  jt != track2->recHitsEnd(); jt++){
	      if ((*jt)->isValid()){
                if (((*it)->geographicalId()==(*jt)->geographicalId())&&((*it)->localPosition().x()==(*jt)->localPosition().x()))noverlap++;
              }
            }
          }
        }
        float fi=float(noverlap)/float(track->recHitsSize()); float fj=float(noverlap)/float(track2->recHitsSize());
        //std::cout << " trk1 trk2 nhits1 nhits2 nover " << i << " " << j << " " << track->recHitsSize() << " "  << track2->recHitsSize() << " " << noverlap << " " << fi << " " << fj  <<std::endl;
        if ((fi>shareFrac)||(fj>shareFrac)){
          if (fi<fj){
            selected1[j]=0; 
            //std::cout << " removing 2nd trk in pair " << std::endl;
          }else{
            if (fi>fj){
              selected1[i]=0; 
              //std::cout << " removing 1st trk in pair " << std::endl;
            }else{
              //std::cout << " removing worst chisq in pair " << std::endl;
              if (track->normalizedChi2() > track2->normalizedChi2()){selected1[i]=0;}else{selected1[j]=0;}
            }//end fi > or = fj
          }//end fi < fj
        }//end got a duplicate
      }//end track2 loop
    }//end track loop
   }//end more than 1 track

  //
  //  L2 has > 1 track - try merging
  //
   if ( 1<tC2.size() ){
    int i=-1;
    for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); track++){
      i++; 
      //std::cout << "Track number "<< i << std::endl ; 
      if (!selected2[i])continue;
      int j=-1;
      for (reco::TrackCollection::const_iterator track2=tC2.begin(); track2!=tC2.end(); track2++){
        j++;
        if ((j<=i)||(!selected2[j])||(!selected2[i]))continue;
        int noverlap=0;
        for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
          if ((*it)->isValid()){
            for (trackingRecHit_iterator jt = track2->recHitsBegin();  jt != track2->recHitsEnd(); jt++){
	      if ((*jt)->isValid()){
                if (((*it)->geographicalId()==(*jt)->geographicalId())&&((*it)->localPosition().x()==(*jt)->localPosition().x()))noverlap++;
              }
            }
          }
        }
        float fi=float(noverlap)/float(track->recHitsSize()); float fj=float(noverlap)/float(track2->recHitsSize());
        //std::cout << " trk1 trk2 nhits1 nhits2 nover " << i << " " << j << " " << track->recHitsSize() << " "  << track2->recHitsSize() << " " << noverlap << " " << fi << " " << fj  <<std::endl;
        if ((fi>shareFrac)||(fj>shareFrac)){
          if (fi<fj){
            selected2[j]=0; 
            //std::cout << " removing 2nd trk in pair " << std::endl;
          }else{
            if (fi>fj){
              selected2[i]=0; 
              //std::cout << " removing 1st trk in pair " << std::endl;
            }else{
              //std::cout << " removing worst chisq in pair " << std::endl;
              if (track->normalizedChi2() > track2->normalizedChi2()){selected2[i]=0;}else{selected2[j]=0;}
            }//end fi > or = fj
          }//end fi < fj
        }//end got a duplicate
      }//end track2 loop
    }//end track loop
   }//end more than 1 track

  //
  //  L1 and L2 both have > 0 track - try merging
  //
   if ( (0<tC1.size())&&(0<tC2.size()) ){
    i=-1;
    for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); track++){
      i++; 
      //std::cout << "L1 Track number "<< i << std::endl ; 
      if (!selected1[i])continue;
      int j=-1;
      for (reco::TrackCollection::const_iterator track2=tC2.begin(); track2!=tC2.end(); track2++){
        j++;
        if ((!selected2[j])||(!selected1[i]))continue;
        int noverlap=0;
        for (trackingRecHit_iterator it = track->recHitsBegin();  it != track->recHitsEnd(); it++){
          if ((*it)->isValid()){
            for (trackingRecHit_iterator jt = track2->recHitsBegin();  jt != track2->recHitsEnd(); jt++){
	      if ((*jt)->isValid()){
                float delta = fabs ( (*it)->localPosition().x()-(*jt)->localPosition().x() ); 
                if (((*it)->geographicalId()==(*jt)->geographicalId())&&(delta<epsilon))noverlap++;
              }
            }
          }
        }
        float fi=float(noverlap)/float(track->recHitsSize()); float fj=float(noverlap)/float(track2->recHitsSize());
        //std::cout << " trk1 trk2 nhits1 nhits2 nover " << i << " " << j << " " << track->recHitsSize() << " "  << track2->recHitsSize() << " " << noverlap << " " << fi << " " << fj  <<std::endl;
        if ((fi>shareFrac)||(fj>shareFrac)){
          if (fi<fj){
            selected2[j]=0; 
            //std::cout << " removing L2 trk in pair " << std::endl;
          }else{
            if (fi>fj){
              selected1[i]=0; 
              //std::cout << " removing L1 trk in pair " << std::endl;
            }else{
              //std::cout << " removing worst chisq in pair " << track->normalizedChi2() << " " << track2->normalizedChi2() << std::endl;
              if (track->normalizedChi2() > track2->normalizedChi2()){selected1[i]=0;}else{selected2[j]=0;}
            }//end fi > or = fj
          }//end fi < fj
        }//end got a duplicate
      }//end track2 loop
    }//end track loop
   }//end more than 1 track

  //
  //  output selected tracks - if any
  //
   if ( 0<tC1.size() ){
    i=-1;
    for (reco::TrackCollection::const_iterator track=tC1.begin(); track!=tC1.end(); track++){
      i++;  if (!selected1[i])continue;
        reco::Track * theTrack = new reco::Track(track->chi2(),
						 (short unsigned)track->ndof(),
						 track->innerPosition(),
						 track->innerMomentum(),
						 track->charge(),
						 track->innerStateCovariance());    
      //fill the TrackCollection
      reco::TrackExtraRef theTrackExtraRef=track->extra();    
      theTrack->setExtra(theTrackExtraRef);    
      theTrack->setHitPattern((*theTrackExtraRef).recHits());
      output->push_back(*theTrack);
      delete theTrack;
    }//end faux loop over tracks
   }//end more than 0 track
   if ( 0<tC2.size() ){
    i=-1;
    for (reco::TrackCollection::const_iterator track=tC2.begin(); track!=tC2.end(); track++){
      i++;  if (!selected2[i])continue;
        reco::Track * theTrack = new reco::Track(track->chi2(),
						 (short unsigned)track->ndof(),
						 track->innerPosition(),
						 track->innerMomentum(),
						 track->charge(),
						 track->innerStateCovariance());    
      //fill the TrackCollection
      reco::TrackExtraRef theTrackExtraRef=track->extra();    
      theTrack->setExtra(theTrackExtraRef);    
      theTrack->setHitPattern((*theTrackExtraRef).recHits());
      output->push_back(*theTrack);
      delete theTrack;
    }//end faux loop over tracks
   }//end more than 0 track


    e.put(output);
    return;

  }//end produce

}
