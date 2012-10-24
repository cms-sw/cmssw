//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker
// Class:           RoadSearchTrackCandidateMakerAlgorithm
// 
// Description:     Converts cleaned clouds into
//                  TrackCandidates using the 
//                  TrajectoryBuilder framework
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 15 13:00:00 UTC 2006
//
// $Author: eulisse $
// $Date: 2012/10/18 09:04:57 $
// $Revision: 1.1 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "RoadSearchTrackCandidateMakerAlgorithm.h"

#include "TrackingTools/RoadSearchHitAccess/interface/RoadSearchHitSorting.h"

#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloud.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedGenerator/interface/FastLine.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

RoadSearchTrackCandidateMakerAlgorithm::RoadSearchTrackCandidateMakerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
  
  theNumHitCut = (unsigned int)conf_.getParameter<int>("NumHitCut");
  theChi2Cut   = conf_.getParameter<double>("HitChi2Cut");
  
  theEstimator = new Chi2MeasurementEstimator(theChi2Cut);
  theUpdator = new KFUpdator();
  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits;
  
  CosmicReco_  = conf_.getParameter<bool>("StraightLineNoBeamSpotCloud");
  CosmicTrackMerging_ = conf_.getParameter<bool>("CosmicTrackMerging");
  MinChunkLength_ = conf_.getParameter<int>("MinimumChunkLength");
  nFoundMin_      = conf_.getParameter<int>("nFoundMin");
  
  initialVertexErrorXY_  = conf_.getParameter<double>("InitialVertexErrorXY");
  splitMatchedHits_  = conf_.getParameter<bool>("SplitMatchedHits");
  cosmicSeedPt_  = conf_.getParameter<double>("CosmicSeedPt");

  measurementTrackerName_ = conf_.getParameter<std::string>("MeasurementTrackerName");
  
  debug_ = false;
  debugCosmics_ = false;

  maxPropagationDistance = 1000.0; // 10m
}

RoadSearchTrackCandidateMakerAlgorithm::~RoadSearchTrackCandidateMakerAlgorithm() {
  delete theEstimator;
  delete theUpdator;
  delete theTrajectoryCleaner;
  // delete theMeasurementTracker;

}

void RoadSearchTrackCandidateMakerAlgorithm::run(const RoadSearchCloudCollection* input,
                                                 const edm::Event& e,
                                                 const edm::EventSetup& es,
                                                 TrackCandidateCollection &output)
{
  
  //
  // right now, track candidates are just filled from cleaned
  // clouds. The trajectory of the seed is taken as the initial
  // trajectory for the final fit
  //
  
  //
  // get the transient builder
  //
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  es.get<TransientRecHitRecord>().get(builderName,theBuilder);
  ttrhBuilder = theBuilder.product();
  
  edm::ESHandle<MeasurementTracker>    measurementTrackerHandle;
  es.get<CkfComponentsRecord>().get(measurementTrackerName_, measurementTrackerHandle);
  theMeasurementTracker = measurementTrackerHandle.product();
  
  std::vector<Trajectory> FinalTrajectories;
  
  
  // need this to sort recHits, sorting done after getting seed because propagationDirection is needed
  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  trackerGeom = tracker.product();
  
  edm::ESHandle<MagneticField> magField_;
  es.get<IdealMagneticFieldRecord>().get(magField_);
  magField = magField_.product();
  
  NoFieldCosmic_ = (CosmicReco_ && (magField->inTesla(GlobalPoint(0,0,0)).mag() < 0.01));

  theMeasurementTracker->update(e);
  //const MeasurementTracker*  theMeasurementTracker = new MeasurementTracker(es,mt_params); // will need this later
  
  theAloPropagator = new PropagatorWithMaterial(alongMomentum,.1057,&(*magField)); 
  theRevPropagator = new PropagatorWithMaterial(oppositeToMomentum,.1057,&(*magField)); 
  theAnalyticalPropagator = new AnalyticalPropagator(magField,anyDirection);

  thePropagator = theAloPropagator;

  //KFTrajectorySmoother theSmoother(*theRevPropagator, *theUpdator, *theEstimator);
  theSmoother = new KFTrajectorySmoother(*theRevPropagator, *theUpdator, *theEstimator);
  
  // get hit matcher
  theHitMatcher = new SiStripRecHitMatcher(3.0);

  //debug_ = true;
  //if (input->size()>0) debug_ = true;

  LogDebug("RoadSearch") << "Clean Clouds input size: " << input->size();
  if (debug_) std::cout << std::endl << std::endl
			<< "*** NEW EVENT: Clean Clouds input size: " << input->size() << std::endl;
  
  int i_c = 0;
  for ( RoadSearchCloudCollection::const_iterator cloud = input->begin(); cloud != input->end(); ++cloud ) {
    
    // fill rechits from cloud into new
    RoadSearchCloud::RecHitVector recHits = cloud->recHits();
    
    std::vector<Trajectory> CloudTrajectories;
    
    if (!CosmicReco_){
      std::sort(recHits.begin(),recHits.end(),SortHitPointersByGlobalPosition(tracker.product(),alongMomentum));
    }
    else {
      std::sort(recHits.begin(),recHits.end(),SortHitPointersByY(*tracker));
    }

    const unsigned int nlost_max = 2;
	    
    // make a list of layers in cloud and mark stereo layers
    
    const unsigned int max_layers = 128;

    // collect hits in cloud by layer
    std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > > RecHitsByLayer;
    std::map<const DetLayer*, int> cloud_layer_reference; // for debugging
    std::map<const DetLayer*, int>::iterator hiter;
    for(RoadSearchCloud::RecHitVector::const_iterator ihit = recHits.begin();
	ihit != recHits.end(); ihit++) {
      // only use useful layers
      const DetLayer* thisLayer =
	theMeasurementTracker->geometricSearchTracker()->detLayer((*ihit)->geographicalId());

      std::map<const DetLayer*, int>::const_iterator ilyr = cloud_layer_reference.find(thisLayer);
      if (ilyr==cloud_layer_reference.end())
	cloud_layer_reference.insert(std::make_pair( thisLayer, RecHitsByLayer.size()));

      if (!RecHitsByLayer.empty() && RecHitsByLayer.back().first == thisLayer) { // Same as previous layer
	RecHitsByLayer.back().second.push_back(*ihit);
      }
      else { // check if this is a new layer
	if (ilyr != cloud_layer_reference.end()){// Not a New Layer
	  int ilayer = ilyr->second;
	  (RecHitsByLayer.begin()+ilayer)->second.push_back(*ihit);
	}
	else{// New Layer
          if (RecHitsByLayer.size() >= max_layers) break; // should never happen
	  lstereo[RecHitsByLayer.size()] = false;
	  if ((*ihit)->localPositionError().yy()<1.) lstereo[RecHitsByLayer.size()] = true;
	  RoadSearchCloud::RecHitVector rhc;
	  rhc.push_back(*ihit);
	  RecHitsByLayer.push_back(std::make_pair(thisLayer, rhc));
	}	
      }
    }

    LogDebug("RoadSearch")<<"Cloud #"<<i_c<<" has "<<recHits.size()<<" hits in "<<RecHitsByLayer.size()<<" layers ";
    if (debug_) std::cout <<"Cloud "<<i_c<<" has "<<recHits.size()<<" hits in " <<RecHitsByLayer.size() << " layers " <<std::endl;;
    ++i_c;

    if (debug_){
      int ntothit = 0;

      for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilhv = RecHitsByLayer.begin();
	   ilhv != RecHitsByLayer.end(); ++ilhv) {
	std::cout<<"   Layer " << ilhv-RecHitsByLayer.begin() << " has " << ilhv->second.size() << " hits " << std::endl;
      }
      std::cout<<std::endl;
      for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilhv = RecHitsByLayer.begin();
	   ilhv != RecHitsByLayer.end(); ++ilhv) {
	RoadSearchCloud::RecHitVector theLayerHits = ilhv->second;
	for (RoadSearchCloud::RecHitVector::const_iterator ihit = theLayerHits.begin();
           ihit != theLayerHits.end(); ++ihit) {
	
	  GlobalPoint gp = trackerGeom->idToDet((*ihit)->geographicalId())->surface().toGlobal((*ihit)->localPosition());
	  if (CosmicReco_){
	    std::cout << "   Hit "<< ntothit
		      << " x/y/z = "
		      << gp.x() << " " << gp.y() << " " << gp.z()
		      <<" in layer " << ilhv-RecHitsByLayer.begin()
		      << " is hit " << (ihit-theLayerHits.begin())+1 
		      << " of " << theLayerHits.size() << std::endl;
	  }
	  else {
	    std::cout << "   Hit "<< ntothit
		      << " r/z = "
		      << gp.perp() << " " << gp.z()
		      <<" in layer " << ilhv-RecHitsByLayer.begin()
		      << " is hit " << (ihit-theLayerHits.begin())+1 
		      << " of " << theLayerHits.size() << std::endl;
	  }
	  ntothit++;
	}
      }
      std::cout<<std::endl;
    }

    // try to start from all layers until the chunk is too short
    //
    
    for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilyr0 = RecHitsByLayer.begin();
	 ilyr0 != RecHitsByLayer.end(); ++ilyr0) {

      unsigned int ilayer0 = (unsigned int)(ilyr0-RecHitsByLayer.begin());
      if (ilayer0 > RecHitsByLayer.size()-MinChunkLength_) continue;      

      std::vector<Trajectory> ChunkTrajectories;
      std::vector<Trajectory> CleanChunks;
      bool all_chunk_layers_used = false;
      
      if (debug_) std::cout  << "*** START NEW CHUNK --> layer range (" << ilyr0-RecHitsByLayer.begin() 
			     << "-" << RecHitsByLayer.size()-1 << ")";

      // collect hits from the starting layer
      RoadSearchCloud::RecHitVector recHits_start = ilyr0->second;

      //
      // Step 1: find small tracks (chunks) made of hits
      // in layers with low occupancy
      //
      
      // find layers with small number of hits
      // TODO: try to keep earliest layers + at least one stereo layer
      std::multimap<int, const DetLayer*> layer_map;
      std::map<const DetLayer*, int> layer_reference; // for debugging
                                                      // skip starting layer, as it is always included
      for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilayer = ilyr0+1;
	   ilayer != RecHitsByLayer.end(); ++ilayer) {
        layer_map.insert(std::make_pair(ilayer->second.size(), ilayer->first));
	layer_reference.insert(std::make_pair(ilayer->first, ilayer-RecHitsByLayer.begin()));
      }
      
      if (debug_) {
	std::cout<<std::endl<<"   Available layers are: " << std::endl;
	for (std::multimap<int, const DetLayer*>::iterator ilm1 = layer_map.begin();
	     ilm1 != layer_map.end(); ++ilm1) {
	  std::map<const DetLayer*, int>::iterator ilr = layer_reference.find(ilm1->second);
	  if (ilr != layer_reference.end() && debug_) 
	    std::cout << "Layer " << ilr->second << " with " << ilm1->first <<" hits" <<std::endl;;
	}
      }

      //const int max_middle_layers = 2;
      std::set<const DetLayer*> the_good_layers;
      std::vector<const DetLayer*> the_middle_layers;
      RoadSearchCloud::RecHitVector the_recHits_middle;

      //      bool StartLayers = 
      chooseStartingLayers(RecHitsByLayer,ilyr0,layer_map,the_good_layers,the_middle_layers,the_recHits_middle);
      if (debug_) {
	std::cout << " From new code... With " << the_good_layers.size() << " useful layers: ";
	for (std::set<const DetLayer*>::iterator igl = the_good_layers.begin();
	     igl!= the_good_layers.end(); ++igl){
	  std::map<const DetLayer*, int>::iterator ilr = layer_reference.find(*igl);
	  if (ilr != layer_reference.end()) std::cout << " " << ilr->second;
	}
	std::cout << std::endl;
	std::cout << " From new code... and middle layers: ";
	for (std::vector<const DetLayer*>::iterator iml = the_middle_layers.begin();
	     iml!= the_middle_layers.end(); ++iml){
	  std::map<const DetLayer*, int>::iterator ilr = layer_reference.find(*iml);
	  if (ilr != layer_reference.end()) std::cout << " " << ilr->second;
	}
	std::cout << std::endl;
      }
      RoadSearchCloud::RecHitVector recHits_inner = recHits_start;
      RoadSearchCloud::RecHitVector recHits_outer = the_recHits_middle;
      std::set<const DetLayer*> good_layers = the_good_layers;
      unsigned int ngoodlayers = good_layers.size();

      if (debug_)
	std::cout<<"Found " << recHits_inner.size() << " inner hits and " << recHits_outer.size() << " outer hits" << std::endl;

      // collect hits in useful layers
      std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > > goodHits;
      // mark layers that will be skipped in first pass
      std::set<const DetLayer*> skipped_layers;
      std::map<int, const DetLayer*> skipped_layer_detmap;

      
      goodHits.push_back(*ilyr0); // save hits from starting layer
      // save hits from other good layers
      for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilyr = ilyr0+1;
	 ilyr != RecHitsByLayer.end(); ++ilyr) {
        if (good_layers.find(ilyr->first) != good_layers.end()){
	  goodHits.push_back(*ilyr);
	}
	else {
          skipped_layers.insert(ilyr->first);
          std::map<const DetLayer*, int>::iterator ilr = layer_reference.find(ilyr->first);
          if (ilr != layer_reference.end())
            skipped_layer_detmap.insert(std::make_pair(ilr->second,ilyr->first));
          else
            if (debug_) std::cout<<"Couldn't find thisLayer to insert into map..."<<std::endl;
	}
      }

      // try various hit combinations
      for (RoadSearchCloud::RecHitVector::const_iterator innerHit = recHits_inner.begin();
           innerHit != recHits_inner.end(); ++innerHit) {

	const DetLayer* innerHitLayer =
	  theMeasurementTracker->geometricSearchTracker()->detLayer((*innerHit)->geographicalId());


	RoadSearchCloud::RecHitVector::iterator middleHit, outerHit;
	RoadSearchCloud::RecHitVector::iterator firstHit, lastHit;

	bool triplets = (CosmicReco_ && (magField->inTesla(GlobalPoint(0,0,0)).mag() > 0.01));
	
	if (!triplets){
	  firstHit = recHits_outer.begin();
	  lastHit  = recHits_outer.end();
	}
	else if (triplets){
	  firstHit = recHits_outer.begin()+1;
	  lastHit = recHits_outer.end();
	}

        for (RoadSearchCloud::RecHitVector::iterator outerHit = firstHit; outerHit != lastHit; ++outerHit) {
          
	  const DetLayer* middleHitLayer = 0;
	  if (triplets){
	    middleHit = outerHit-1;
	    middleHitLayer = theMeasurementTracker->geometricSearchTracker()->detLayer((*middleHit)->geographicalId());
	  }
	  const DetLayer* outerHitLayer =
	    theMeasurementTracker->geometricSearchTracker()->detLayer((*outerHit)->geographicalId());
	  if (middleHitLayer == outerHitLayer) continue;

	  FreeTrajectoryState fts;
	  if (!triplets){
	    if (debug_){
	      std::map<const DetLayer*, int>::iterator ilro = layer_reference.find(outerHitLayer);
	      if (ilro != layer_reference.end()) {
		std::cout << "Try trajectory with Inner Hit on Layer " << ilayer0 << " and  " ;
		std::cout << "Outer Hit on Layer " << ilro->second << std::endl;
	      }
	    }
	    fts = initialTrajectory(es,*innerHit,*outerHit);
	  }
	  else if (triplets){
	    if (debug_){
	      std::map<const DetLayer*, int>::iterator ilrm = layer_reference.find(middleHitLayer);
	      std::map<const DetLayer*, int>::iterator ilro = layer_reference.find(outerHitLayer);
	      if (ilro != layer_reference.end() && ilrm != layer_reference.end()) {
		std::cout << "Try trajectory with Hits on Layers " << ilayer0 << " , "
			  << ilrm->second <<  " and  " << ilro->second << std::endl;
	      }
	    }
	    fts = initialTrajectoryFromTriplet(es,*innerHit,*middleHit,*outerHit);
	  }

	  if (!fts.hasError()) continue;
	  if (debug_) std::cout<<"FTS: " << fts << std::endl;

	  Trajectory seedTraj = createSeedTrajectory(fts,*innerHit,innerHitLayer);

          std::vector<Trajectory> rawTrajectories;          
	  if (seedTraj.isValid() && !seedTraj.measurements().empty() ) rawTrajectories.push_back(seedTraj);//GC
          //rawTrajectories.push_back(seedTraj);

	  int ntested = 0;
          // now loop on hits
          std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilyr_start = (goodHits.begin()+1);
          for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilhv = ilyr_start;
               ilhv != goodHits.end(); ++ilhv) {
            RoadSearchCloud::RecHitVector& hits = ilhv->second;
            //std::vector<Trajectory> newTrajectories;
	    ++ntested;
            if (debug_){
              std::map<const DetLayer*, int>::iterator ilr = cloud_layer_reference.find(ilhv->first);
              if (ilr != cloud_layer_reference.end())
                std::cout << "extrapolating " << rawTrajectories.size() 
			  << " trajectories to layer " << ilr->second 
			  << " which has  " << hits.size() << " hits " << std::endl;
            }
            
	    std::vector<Trajectory>newTrajectories;
	    for (std::vector<Trajectory>::const_iterator it = rawTrajectories.begin();
		     it != rawTrajectories.end(); it++) {
	      if (debug_) std::cout << "extrapolating Trajectory #" << it-rawTrajectories.begin() << std::endl;
	      if (it->direction()==alongMomentum) thePropagator = theAloPropagator;//GC
	      else thePropagator = theRevPropagator;

	      std::vector<Trajectory> theTrajectories = extrapolateTrajectory(*it,hits,
									      innerHitLayer, *outerHit, outerHitLayer);
	      if (theTrajectories.empty()) {
		if (debug_) std::cout<<" Could not add the hit in this layer " << std::endl;
		if (debug_){
		  std::cout << " --> trajectory " << it-rawTrajectories.begin() 
			    << " has "<<it->recHits().size()<<" hits after "
			    << (ilhv-ilyr_start+1) << " tested (ntested=" <<ntested<<") "
			    << " --> misses="<< (ilhv-ilyr_start+1)-(it->recHits().size()-1)
			    << " but there are " << (goodHits.end() - ilhv)
			    <<" more layers in first pass and "<< skipped_layers.size() <<" skipped layers " <<std::endl;
		  
		  
		}
		// layer missed
		if ((ilhv-ilyr_start+1)-(it->recHits().size()-1) <= nlost_max){
		  newTrajectories.push_back(*it);
		}
	      }
	      else{ // added hits in this layers
		for (std::vector<Trajectory>::const_iterator it = theTrajectories.begin();
		     it != theTrajectories.end(); it++) {
		  newTrajectories.push_back(*it);
		}
	      }
            } // end loop over rawTrajectories
            rawTrajectories = newTrajectories;
	    if (newTrajectories.empty()) break;
	  }
          if (rawTrajectories.size()==0){
	    continue;
	    if (debug_) std::cout<<" --> yields ZERO raw trajectories!" << std::endl;
	  }
	  if (debug_){
	    for (std::vector<Trajectory>::const_iterator it = rawTrajectories.begin();
		 it != rawTrajectories.end(); it++) {
	      std::cout << " --> yields trajectory with "<<it->recHits().size()<<" hits with chi2="
			<<it->chiSquared()<<" and is valid? "<<it->isValid() <<std::endl;
	    }
	  }
          std::vector<Trajectory> rawCleaned;
          theTrajectoryCleaner->clean(rawTrajectories);
          for (std::vector<Trajectory>::const_iterator itr = rawTrajectories.begin();
               itr != rawTrajectories.end(); ++itr) {
            // see how many layers have been found
            if (!itr->isValid()) continue;
            std::set<const DetLayer*> used_layers;
            Trajectory::DataContainer tmv = itr->measurements();
            for (Trajectory::DataContainer::iterator itm = tmv.begin();
                 itm != tmv.end(); ++itm) {
              TransientTrackingRecHit::ConstRecHitPointer rh = itm->recHit();
              if (!rh->isValid()) continue;
              used_layers.insert(theMeasurementTracker->geometricSearchTracker()->detLayer(rh->geographicalId()));
            }

	    // need to subtract 1 from used_layers since it includes the starting layer
	    if (debug_) std::cout<<"Used " << (used_layers.size()-1) << " layers out of " << ngoodlayers
				 << " good layers, so " << ngoodlayers - (used_layers.size()-1) << " missed "
				 << std::endl;
            if ((int)used_layers.size() < nFoundMin_) continue;
            unsigned int nlostlayers = ngoodlayers - (used_layers.size()-1);
            if (nlostlayers > nlost_max) continue;
            
            rawCleaned.push_back(*itr);
            
          }
          if (!rawCleaned.empty()) {
            ChunkTrajectories.insert(ChunkTrajectories.end(), rawCleaned.begin(), rawCleaned.end());
          }
	} // END LOOP OVER OUTER HITS
      } // END LOOP OVER INNER HITS
      // At this point we have made all the trajectories from the low occupancy layers
      // We clean these trajectories first, and then try to add hits from the skipped layers
      
      //    }
      if (debug_) std::cout << "Clean the " << ChunkTrajectories.size()<<" trajectories for this chunk" << std::endl;
      // clean the intermediate result
      theTrajectoryCleaner->clean(ChunkTrajectories);
      for (std::vector<Trajectory>::const_iterator it = ChunkTrajectories.begin();
           it != ChunkTrajectories.end(); it++) {
        if (it->isValid())  CleanChunks.push_back(*it);
      }
      if (debug_) std::cout <<"After cleaning there are " << CleanChunks.size() << " trajectories for this chunk" << std::endl;

      
      // *********************  BEGIN NEW ADDITION      
      
      //
      // Step 2: recover measurements from busy layers
      //
      
      std::vector<Trajectory> extendedChunks;
      

      // see if there are layers that we skipped

      if (debug_){
        if (skipped_layers.empty()) {
          std::cout << "all layers were used in first pass" << std::endl;
        } else {
          std::cout << "There are " << skipped_layer_detmap.size() << " skipped layers:";
          for (std::map<int, const DetLayer*>::const_iterator imap = skipped_layer_detmap.begin();
               imap!=skipped_layer_detmap.end(); imap++){
            std::cout<< " " <<imap->first;
          }
          std::cout << std::endl;
        }
      }
      
      for (std::vector<Trajectory>::const_iterator i = CleanChunks.begin();
           i != CleanChunks.end(); i++) {
        if (!(*i).isValid()) continue;
        if (debug_) std::cout<< "Now process CleanChunk trajectory " << i-CleanChunks.begin() << std::endl;
	bool all_layers_used = false;
        if (skipped_layers.empty() && i->measurements().size() >= theNumHitCut) {
	  if (debug_) std::cout<<"The trajectory has " << i->measurements().size() 
			       << " hits from a cloud of " << RecHitsByLayer.size() 
			       << " layers and a chunk of " << (RecHitsByLayer.size() - ilayer0) << " layers " << std::endl;
          extendedChunks.insert(extendedChunks.end(), *i);
	  if (i->measurements().size() >= (RecHitsByLayer.size() - ilayer0)){
	    all_layers_used = true;
	    break;
	  }
        } 
        else {
          
          Trajectory temWptraj = *i;
          Trajectory::DataContainer tmv = (*i).measurements();
          if (tmv.size()+skipped_layer_detmap.size() < theNumHitCut) continue;          

	  // Debug dump of hits
	  if (debug_){
	    for (Trajectory::DataContainer::const_iterator ih=tmv.begin();
		 ih!=tmv.end();++ih){
	      TransientTrackingRecHit::ConstRecHitPointer rh = ih->recHit();
	      const DetLayer* Layer =
		theMeasurementTracker->geometricSearchTracker()->detLayer(rh->geographicalId());      
	      std::map<const DetLayer*, int>::iterator ilr = cloud_layer_reference.find(Layer);
	      if (ilr != cloud_layer_reference.end())
		std::cout << "   Hit #"<<ih-tmv.begin() << " of " << tmv.size()
			  <<" is on Layer " << ilr->second << std::endl;
	      else 
		std::cout << "   Layer for Hit #"<<ih-tmv.begin() <<" can't be found " << std::endl;
	      std::cout<<" updatedState:\n" << ih->updatedState() << std::endl;
	      std::cout<<" predictState:\n" << ih->predictedState() << std::endl;
	    }
	  }
          
          // Loop over the layers in the cloud
          
	  std::set<const DetLayer*> final_layers;
          Trajectory::DataContainer::const_iterator im = tmv.begin();
          Trajectory::DataContainer::const_iterator im2 = tmv.begin();
          
          TrajectoryMeasurement firstMeasurement = i->firstMeasurement();
          const DetLayer* firstDetLayer = 
            theMeasurementTracker->geometricSearchTracker()->detLayer(firstMeasurement.recHit()->geographicalId());
          
          std::vector<Trajectory> freshStartv = theSmoother->trajectories(*i);
          
          if(debug_) std::cout<<"Smoothing has returned " << freshStartv.size() <<" trajectory " << std::endl;
          if (!freshStartv.empty()){
             if(debug_)  std::cout<< "Smoothing of trajectory " <<i-CleanChunks.begin() << " has succeeded with " << freshStartv.begin()->measurements().size() << " hits and a chi2 of " << freshStartv.begin()->chiSquared() <<" for " << freshStartv.begin()->ndof() << " DOF.  Now add hits." <<std::endl;
          } else {
             if (debug_) std::cout<< "Smoothing of trajectory " <<i-CleanChunks.begin() <<" has failed"<<std::endl;
             continue;
          }
          
          Trajectory freshStart = *freshStartv.begin();
          std::vector<TrajectoryMeasurement> freshStartTM = freshStart.measurements();
          
          if (debug_) {
             for (std::vector<TrajectoryMeasurement>::const_iterator itm = freshStartTM.begin();itm != freshStartTM.end(); ++itm){
                std::cout<<"Trajectory hit " << itm-freshStartTM.begin() <<" updatedState:\n" << itm->updatedState() << std::endl;
             }
          }

          TrajectoryStateOnSurface NewFirstTsos = freshStart.lastMeasurement().updatedState();
          if(debug_) std::cout<<"NewFirstTSOS is valid? " << NewFirstTsos.isValid() << std::endl;
          if(debug_) std::cout << " NewFirstTSOS:\n " << NewFirstTsos << std::endl;
          TransientTrackingRecHit::ConstRecHitPointer rh = freshStart.lastMeasurement().recHit();

          if(debug_) {
             std::cout<< "First hit for fresh start on det " << rh->det() << ", r/phi/z = " << rh->globalPosition().perp() << " " << rh->globalPosition().phi() << " " << rh->globalPosition().z();
          }
          
          PTrajectoryStateOnDet const & pFirstState = trajectoryStateTransform::persistentState(NewFirstTsos,
                                                                                          rh->geographicalId().rawId());
          edm::OwnVector<TrackingRecHit> newHits;
          newHits.push_back(rh->hit()->clone());
          
          TrajectorySeed tmpseed = TrajectorySeed(pFirstState, 
                                                  newHits,
                                                  i->direction());

	  thePropagator = theAloPropagator;
	  if (i->direction()==oppositeToMomentum) thePropagator = theRevPropagator;

          
          
          Trajectory newTrajectory(tmpseed,tmpseed.direction());
          
          const GeomDet* det = trackerGeom->idToDet(rh->geographicalId());
          TrajectoryStateOnSurface invalidState(new BasicSingleTrajectoryState(det->surface()));
          newTrajectory.push(TrajectoryMeasurement(invalidState, NewFirstTsos, rh, 0, firstDetLayer));
	  final_layers.insert(firstDetLayer);

	  if(debug_) std::cout << "TRAJ is valid: " << newTrajectory.isValid() <<std::endl;

	  TrajectoryStateOnSurface testTsos = newTrajectory.measurements().back().updatedState();
	  
	  if(debug_) {
	    std::cout << "testTSOS is valid!!" << testTsos.isValid() << std::endl;
	    std::cout << " testTSOS (x/y/z): " << testTsos.globalPosition().x()<< " / " << testTsos.globalPosition().y()<< " / " << testTsos.globalPosition().z() << std::endl;
	    std::cout << " local position x: " << testTsos.localPosition().x() << "+-" << sqrt(testTsos.localError().positionError().xx()) << std::endl;
	  }

	  if (firstDetLayer != ilyr0->first){
	    if (debug_) std::cout<<"!!! ERROR !!! firstDetLayer ("<<firstDetLayer<<") != ilyr0 ( " <<ilyr0->first <<")"<< std::endl;
	  }          
	  ++im;

	  if (debug_){
	    std::map<const DetLayer*, int>::iterator ilr = cloud_layer_reference.find(firstDetLayer);
	    if (ilr != cloud_layer_reference.end() ){
	      std::cout << "   First hit is on layer  " << ilr->second << std::endl;
	    }
	  }
	  for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilyr = ilyr0+1;
	       ilyr != RecHitsByLayer.end(); ++ilyr) {

            TrajectoryStateOnSurface predTsos;
            TrajectoryStateOnSurface currTsos;
            TrajectoryMeasurement tm;

	    if(debug_)
	      std::cout<<"Trajectory has " << newTrajectory.measurements().size() << " measurements with " << (RecHitsByLayer.end()-ilyr)
		       << " remaining layers " << std::endl;

	    if (im != tmv.end()) im2 = im;
	    TransientTrackingRecHit::ConstRecHitPointer rh = im2->recHit(); // Current 
	    if (rh->isValid() && 
		theMeasurementTracker->geometricSearchTracker()->detLayer(rh->geographicalId()) == ilyr->first) {
	      
		if (debug_) std::cout<<"   Layer " << ilyr-RecHitsByLayer.begin() <<" has a good hit " << std::endl;
                ++im;
                
                const GeomDet* det = trackerGeom->idToDet(rh->geographicalId());

                currTsos = newTrajectory.measurements().back().updatedState();
                predTsos = thePropagator->propagate(currTsos, det->surface());
                if (!predTsos.isValid()) continue;
		GlobalVector propagationDistance = predTsos.globalPosition() - currTsos.globalPosition();
		if (propagationDistance.mag() > maxPropagationDistance) continue;
                MeasurementEstimator::HitReturnType est = theEstimator->estimate(predTsos, *rh);
		if(debug_) {
		  std::cout << "Propagation distance2 is " << propagationDistance.mag() << std::endl;
		  std::cout << "predTSOS is valid!!" << std::endl;
		  std::cout << " predTSOS (x/y/z): " << predTsos.globalPosition().x()<< " / " << predTsos.globalPosition().y()<< " / " << predTsos.globalPosition().z() << std::endl;
		  std::cout << " local position x: " << predTsos.localPosition().x() << "+-" << sqrt(predTsos.localError().positionError().xx()) << std::endl;
		  std::cout << " local position y: " << predTsos.localPosition().y() << "+-" << sqrt(predTsos.localError().positionError().yy()) << std::endl;
		  std::cout << "currTSOS is valid!! " << currTsos.isValid() <<  std::endl;
		  std::cout << " currTSOS (x/y/z): " << currTsos.globalPosition().x()<< " / " << currTsos.globalPosition().y()<< " / " << currTsos.globalPosition().z() << std::endl;
		  std::cout << " local position x: " << currTsos.localPosition().x() << "+-" << sqrt(currTsos.localError().positionError().xx()) << std::endl;
		  std::cout << " local position y: " << currTsos.localPosition().y() << "+-" << sqrt(currTsos.localError().positionError().yy()) << std::endl;
		}

                if (!est.first) {
                  if (debug_) std::cout<<"Failed to add one of the original hits on a low occupancy layer!!!!" << std::endl;
                  continue;
                }
                currTsos = theUpdator->update(predTsos, *rh);
                tm = TrajectoryMeasurement(predTsos, currTsos, &(*rh),est.second,ilyr->first);

		const TrajectoryStateOnSurface theTSOS = newTrajectory.lastMeasurement().updatedState();
		
		//std::cout << "11TsosBefore (x/y/z): " << theTSOS.globalPosition().x()<< " / " << theTSOS.globalPosition().y()<< " / " << theTSOS.globalPosition().z() << std::endl;
		//std::cout << " 11local position x: " << theTSOS.localPosition().x() << "+-" << sqrt(theTSOS.localError().positionError().xx()) << std::endl;
		//std::cout << " 11local position y: " << theTSOS.localPosition().y() << "+-" << sqrt(theTSOS.localError().positionError().yy()) << std::endl;



                newTrajectory.push(tm,est.second);
		final_layers.insert(ilyr->first);                
            }	    
            else{
              if (debug_) std::cout<<"   Layer " << ilyr-RecHitsByLayer.begin() <<" is one of the skipped layers " << std::endl;
              
              //collect hits in the skipped layer
              edm::OwnVector<TrackingRecHit> skipped_hits;
              std::set<const GeomDet*> dets;
              for (RoadSearchCloud::RecHitVector::const_iterator ih = ilyr->second.begin();
                   ih != ilyr->second.end(); ++ih) {
		skipped_hits.push_back((*ih)->clone());
		dets.insert(trackerGeom->idToDet((*ih)->geographicalId()));
              }
              
	      if(debug_){
		std::cout<<"   ---> probing missing hits (nh="<< skipped_hits.size() << ", nd=" << dets.size() 
			 << ")  in layer " <<  ilyr-RecHitsByLayer.begin() <<std::endl;
              }

              const TrajectoryStateOnSurface theTSOS = newTrajectory.lastMeasurement().updatedState();
	      
	      //std::cout << "TsosBefore (x/y/z): " << theTSOS.globalPosition().x()<< " / " << theTSOS.globalPosition().y()<< " / " << theTSOS.globalPosition().z() << std::endl;
	      //std::cout << " local position x: " << theTSOS.localPosition().x() << "+-" << sqrt(theTSOS.localError().positionError().xx()) << std::endl;
	      //std::cout << " local position y: " << theTSOS.localPosition().y() << "+-" << sqrt(theTSOS.localError().positionError().yy()) << std::endl;

              std::vector<TrajectoryMeasurement> theGoodHits = FindBestHits(theTSOS,dets,theHitMatcher,skipped_hits);
              if (!theGoodHits.empty()){
		final_layers.insert(ilyr->first);
                if (debug_) std::cout<<"Found " << theGoodHits.size() << " good hits to add" << std::endl;
                for (std::vector<TrajectoryMeasurement>::const_iterator im=theGoodHits.begin();im!=theGoodHits.end();++im){
                  newTrajectory.push(*im,im->estimate());
                }
              }
            }
          } // END 2nd PASS LOOP OVER LAYERS IN CLOUD
          
          if (debug_) std::cout<<"Finished loop over layers in cloud.  Trajectory now has " <<newTrajectory.measurements().size()
			       << " hits. " << std::endl;
	  if (debug_) std::cout<<"The trajectory has " << newTrajectory.measurements().size() <<" hits on " << final_layers.size()
			       << " layers from a cloud of " << RecHitsByLayer.size() 
			       << " layers and a chunk of " << (RecHitsByLayer.size() - ilayer0) << " layers " << std::endl;
          if (newTrajectory.measurements().size() >= theNumHitCut)
	    extendedChunks.insert(extendedChunks.end(), newTrajectory);
	  if (final_layers.size() >= (RecHitsByLayer.size() - ilayer0)){
	    if (debug_) std::cout<<"All layers of the chunk have been used..." << std::endl;
	    all_layers_used = true;
	  }
        }  // END ELSE TO RECOVER SKIPPED LAYERS
	if (all_layers_used) {
	  if (debug_) std::cout << "All layers were used, so break....." << std::endl;
	  all_chunk_layers_used = true;
	  break;
	}
	if (debug_) std::cout<<"Going to next clean chunk....." << std::endl;
      } // END LOOP OVER CLEAN CHUNKS
      if (debug_) std::cout<< "Now Clean the " << extendedChunks.size() << " extended chunks " <<std::endl;
      if (extendedChunks.size() > 1) theTrajectoryCleaner->clean(extendedChunks);
      for (std::vector<Trajectory>::const_iterator it = extendedChunks.begin();
           it != extendedChunks.end(); it++) {
        if (it->isValid()) CloudTrajectories.push_back(*it);
      }
      if (all_chunk_layers_used) break;
    }
    
    // ********************* END NEW ADDITION
    
    if (debug_) std::cout<< "Finished with the cloud, so clean the " 
			 << CloudTrajectories.size() << " cloud trajectories "<<std::endl ;
    theTrajectoryCleaner->clean(CloudTrajectories);
    for (std::vector<Trajectory>::const_iterator it = CloudTrajectories.begin();
         it != CloudTrajectories.end(); it++) {
      if (it->isValid()) FinalTrajectories.push_back(*it);
    }
    
    if (debug_) std::cout<<" Final trajectories now has size " << FinalTrajectories.size()<<std::endl ;
    
  } // End loop over Cloud Collection


  if (debug_) std::cout<< " Finished loop over all clouds " <<std::endl;

  output = PrepareTrackCandidates(FinalTrajectories);

  delete theAloPropagator;
  delete theRevPropagator; 
  delete theAnalyticalPropagator;
  delete theHitMatcher;
  delete theSmoother;
  
  if (debug_ || debugCosmics_) std::cout<< "Found " << output.size() << " track candidate(s)."<<std::endl;

}


//edm::OwnVector<TrackingRecHit> 
std::vector<TrajectoryMeasurement>
RoadSearchTrackCandidateMakerAlgorithm::FindBestHitsByDet(const TrajectoryStateOnSurface& tsosBefore,
                                                          const std::set<const GeomDet*>& theDets,
                                                          edm::OwnVector<TrackingRecHit>& theHits)
//			 edm::OwnVector<TrackingRecHit> *theBestHits)
{
  
  //edm::OwnVector<TrackingRecHit> theBestHits;
  std::vector<TrajectoryMeasurement> theBestHits;
  
  // extrapolate to all detectors from the list
  std::map<const GeomDet*, TrajectoryStateOnSurface> dmmap;
  for (std::set<const GeomDet*>::iterator idet = theDets.begin();
       idet != theDets.end(); ++idet) {
    TrajectoryStateOnSurface predTsos = thePropagator->propagate(tsosBefore, (**idet).surface());
    if (predTsos.isValid()) {
      GlobalVector propagationDistance = predTsos.globalPosition() - tsosBefore.globalPosition();
      if (propagationDistance.mag() > maxPropagationDistance) continue; 
      dmmap.insert(std::make_pair(*idet, predTsos));
    }
  }
  // evaluate hit residuals
  std::map<const GeomDet*, TrajectoryMeasurement> dtmmap;
  for (edm::OwnVector<TrackingRecHit>::const_iterator ih = theHits.begin();
       ih != theHits.end(); ++ih) {
    const GeomDet* det = trackerGeom->idToDet(ih->geographicalId());
    
    std::map<const GeomDet*, TrajectoryStateOnSurface>::iterator idm = dmmap.find(det);
    if (idm == dmmap.end()) continue;
    TrajectoryStateOnSurface predTsos = idm->second;
    TransientTrackingRecHit::RecHitPointer rhit = ttrhBuilder->build(&(*ih));
    MeasurementEstimator::HitReturnType est = theEstimator->estimate(predTsos, *rhit);
    
    // Take the best hit on a given Det
    if (est.first) {
      TrajectoryStateOnSurface currTsos = theUpdator->update(predTsos, *rhit);
      std::map<const GeomDet*, TrajectoryMeasurement>::iterator idtm = dtmmap.find(det);
      if (idtm == dtmmap.end()) {
        TrajectoryMeasurement tm(predTsos, currTsos, &(*rhit),est.second,
                                 theMeasurementTracker->geometricSearchTracker()->detLayer(ih->geographicalId()));
        dtmmap.insert(std::make_pair(det, tm));
      } else if (idtm->second.estimate() > est.second) {
        dtmmap.erase(idtm);
        TrajectoryMeasurement tm(predTsos, currTsos, &(*rhit),est.second,
                                 theMeasurementTracker->geometricSearchTracker()->detLayer(ih->geographicalId()));
        dtmmap.insert(std::make_pair(det, tm));
      }
    }
  }

  if (!dtmmap.empty()) {
    for (std::map<const GeomDet*, TrajectoryMeasurement>::iterator idtm = dtmmap.begin();
         idtm != dtmmap.end(); ++idtm) {
      TrajectoryMeasurement itm = idtm->second;
      theBestHits.push_back(itm);
    }
  }
  
  return theBestHits;
}


std::vector<TrajectoryMeasurement>
RoadSearchTrackCandidateMakerAlgorithm::FindBestHit(const TrajectoryStateOnSurface& tsosBefore,
                                                    const std::set<const GeomDet*>& theDets,
                                                    edm::OwnVector<TrackingRecHit>& theHits)
{
  
  std::vector<TrajectoryMeasurement> theBestHits;
  
  double bestchi = 10000.0;
  // extrapolate to all detectors from the list
  std::map<const GeomDet*, TrajectoryStateOnSurface> dmmap;
  for (std::set<const GeomDet*>::iterator idet = theDets.begin();
       idet != theDets.end(); ++idet) {
    TrajectoryStateOnSurface predTsos = thePropagator->propagate(tsosBefore, (**idet).surface());
    if (predTsos.isValid()) {
      GlobalVector propagationDistance = predTsos.globalPosition() - tsosBefore.globalPosition();
      if (propagationDistance.mag() > maxPropagationDistance) continue; 
      dmmap.insert(std::make_pair(*idet, predTsos));
    }
  }
  // evaluate hit residuals
  std::map<const GeomDet*, TrajectoryMeasurement> dtmmap;
  for (edm::OwnVector<TrackingRecHit>::const_iterator ih = theHits.begin();
       ih != theHits.end(); ++ih) {
    const GeomDet* det = trackerGeom->idToDet(ih->geographicalId());
    
    std::map<const GeomDet*, TrajectoryStateOnSurface>::iterator idm = dmmap.find(det);
    if (idm == dmmap.end()) continue;
    TrajectoryStateOnSurface predTsos = idm->second;
    TransientTrackingRecHit::RecHitPointer rhit = ttrhBuilder->build(&(*ih));
    MeasurementEstimator::HitReturnType est = theEstimator->estimate(predTsos, *rhit);
    
    // Take the best hit on any Det
    if (est.first) {
      TrajectoryStateOnSurface currTsos = theUpdator->update(predTsos, *rhit);
      if (est.second < bestchi){
        if(!theBestHits.empty()){
          theBestHits.erase(theBestHits.begin());
        }
        bestchi = est.second;
        TrajectoryMeasurement tm(predTsos, currTsos, &(*rhit),est.second,
                                 theMeasurementTracker->geometricSearchTracker()->detLayer(ih->geographicalId()));	
        theBestHits.push_back(tm);
      }
    }
  }
  
  if (theBestHits.empty()){
    if (debug_) std::cout<< "no hits to add! " <<std::endl;
  }
  else{
    for (std::vector<TrajectoryMeasurement>::const_iterator im=theBestHits.begin();im!=theBestHits.end();++im)
      if (debug_) std::cout<<" Measurement on layer "
			   << theMeasurementTracker->geometricSearchTracker()->detLayer(im->recHit()->geographicalId())
			   << " with estimate " << im->estimate()<<std::endl ;
  }
  
  return theBestHits;
}

std::vector<TrajectoryMeasurement>
RoadSearchTrackCandidateMakerAlgorithm::FindBestHits(const TrajectoryStateOnSurface& tsosBefore,
                                                     const std::set<const GeomDet*>& theDets,
 						     const SiStripRecHitMatcher* theHitMatcher,
                                                     edm::OwnVector<TrackingRecHit>& theHits)
//			 edm::OwnVector<TrackingRecHit> *theBestHits)
{
  

  std::vector<TrajectoryMeasurement> theBestHits;
  //TrajectoryMeasurement* theBestTM = 0;
  TrajectoryMeasurement theBestTM;
  bool firstTM = true;
  
  // extrapolate to all detectors from the list
  std::map<const GeomDet*, TrajectoryStateOnSurface> dmmap;
  for (std::set<const GeomDet*>::iterator idet = theDets.begin();
       idet != theDets.end(); ++idet) {
    TrajectoryStateOnSurface predTsos = thePropagator->propagate(tsosBefore, (**idet).surface());
    if (predTsos.isValid()) {
      GlobalVector propagationDistance = predTsos.globalPosition() - tsosBefore.globalPosition();
      if (propagationDistance.mag() > maxPropagationDistance) continue; 
      dmmap.insert(std::make_pair(*idet, predTsos));
    }
  }

  if(debug_) std::cout << "TRAJECTORY INTERSECTS " << dmmap.size() << " DETECTORS." << std::endl;

  // evaluate hit residuals
  std::map<const GeomDet*, TrajectoryMeasurement> dtmmap;
  for (edm::OwnVector<TrackingRecHit>::const_iterator ih = theHits.begin(); ih != theHits.end(); ++ih) {
    const GeomDet* det = trackerGeom->idToDet(ih->geographicalId());
    //if (*isl != theMeasurementTracker->geometricSearchTracker()->detLayer(ih->geographicalId())) 
    //  std::cout <<" You don't know what you're doing !!!!" << std::endl;
    
    std::map<const GeomDet*, TrajectoryStateOnSurface>::iterator idm = dmmap.find(det);
    if (idm == dmmap.end()) continue;
    TrajectoryStateOnSurface predTsos = idm->second;
    TransientTrackingRecHit::RecHitPointer rhit = ttrhBuilder->build(&(*ih));
    
    const SiStripMatchedRecHit2D *origHit = dynamic_cast<const SiStripMatchedRecHit2D *>(&(*ih));
    if (origHit !=0){
      const GluedGeomDet *gdet = dynamic_cast<const GluedGeomDet*>(det);
      const SiStripMatchedRecHit2D *corrHit = theHitMatcher->match(origHit,gdet,predTsos.localDirection());
      if (corrHit!=0){
	rhit = ttrhBuilder->build(&(*corrHit));
	delete corrHit;
      }
    }

    if (debug_) {
      std::cout << "predTSOS (x/y/z): " << predTsos.globalPosition().x()<< " / " << predTsos.globalPosition().y()<< " / " << predTsos.globalPosition().z() << std::endl;
      std::cout << "local position x: " << predTsos.localPosition().x() << "+-" << sqrt(predTsos.localError().positionError().xx()) << std::endl;
      std::cout << "local position y: " << predTsos.localPosition().y() << "+-" << sqrt(predTsos.localError().positionError().yy()) << std::endl;
      std::cout << "rhit local position x: " << rhit->localPosition().x() << "+-" << sqrt(rhit->localPositionError().xx()) << std::endl;
      std::cout << "rhit local position y: " << rhit->localPosition().y() << "+-" << sqrt(rhit->localPositionError().yy()) << std::endl;
    }
 
    MeasurementEstimator::HitReturnType est = theEstimator->estimate(predTsos, *rhit);
    if (debug_) std::cout<< "hit " << ih-theHits.begin() 
			 << ": est = " << est.first << " " << est.second  <<std::endl;
    
    
    // Take the best hit on a given Det
    if (est.first) {
      TrajectoryMeasurement tm;
      TrajectoryStateOnSurface currTsos = theUpdator->update(predTsos, *rhit);
      std::map<const GeomDet*, TrajectoryMeasurement>::iterator idtm = dtmmap.find(det);
      if (idtm == dtmmap.end()) {
        tm = TrajectoryMeasurement (predTsos, currTsos, &(*rhit),est.second,
                                    theMeasurementTracker->geometricSearchTracker()->detLayer(ih->geographicalId()));
        dtmmap.insert(std::make_pair(det, tm));
      } else if (idtm->second.estimate() > est.second) {
        dtmmap.erase(idtm);
        tm = TrajectoryMeasurement(predTsos, currTsos, &(*rhit),est.second,
                                   theMeasurementTracker->geometricSearchTracker()->detLayer(ih->geographicalId()));
        dtmmap.insert(std::make_pair(det, tm));
      }
      if ((firstTM)){
        theBestTM = tm;	
        if (debug_) std::cout <<"Initialize best to " << theBestTM.estimate() << std::endl;
        firstTM = false;
      }
      else if (!firstTM) {
        if (debug_) std::cout << "Current best is " << theBestTM.estimate() << " while this hit is " << est.second;
        if (est.second < theBestTM.estimate()) {
          if (debug_) std::cout << " so replace it " ;
          theBestTM = tm;
        }
        if (debug_) std::cout << std::endl;
      }
    }
  }
  if (debug_) std::cout<< "Hits(Dets) to add: " << dtmmap.size() <<std::endl;
  if (!dtmmap.empty()) {
    
    std::vector<std::pair<TransientTrackingRecHit::ConstRecHitPointer, TrajectoryMeasurement*> > OverlapHits;
    for (std::map<const GeomDet*, TrajectoryMeasurement>::iterator idtm = dtmmap.begin();
         idtm != dtmmap.end(); ++idtm) {
      OverlapHits.push_back(std::make_pair(idtm->second.recHit(),&idtm->second));
      
      if (debug_) std::cout<<" Measurement on layer "
			   << theMeasurementTracker->geometricSearchTracker()->detLayer(idtm->second.recHit()->geographicalId())
			   << " with estimate " << idtm->second.estimate()<<std::endl ;
    }
    if (debug_)
      std::cout<<" Best  Measurement is on layer "
	       << theMeasurementTracker->geometricSearchTracker()->detLayer(theBestTM.recHit()->geographicalId())
	       << " with estimate " << theBestTM.estimate()<<std::endl ;
    
    
    if (dtmmap.size()==1){  // only one hit so we can just return that one
      for (std::map<const GeomDet*, TrajectoryMeasurement>::iterator idtm = dtmmap.begin();
           idtm != dtmmap.end(); ++idtm) {
        TrajectoryMeasurement itm = idtm->second;
        if (debug_) std::cout<<" Measurement on layer "
			     << theMeasurementTracker->geometricSearchTracker()->detLayer(itm.recHit()->geographicalId())
			     << " with estimate " << itm.estimate()<<std::endl ;
        theBestHits.push_back(itm);
      }
    }
    else if (dtmmap.size()>=2) { // try for the overlaps -- first have to sort inside out
      
      if (debug_) std::cout<<"Unsorted OverlapHits has size " <<OverlapHits.size() << std::endl;
      
      for (std::vector<std::pair<TransientTrackingRecHit::ConstRecHitPointer,TrajectoryMeasurement*> >::iterator irh =OverlapHits.begin();
           irh!=OverlapHits.end();++irh){
        if (debug_) std::cout << "Hit " << irh-OverlapHits.begin()
			      << " on det " << irh->first->det() 
			      << " detLayer " 
			      << theMeasurementTracker->geometricSearchTracker()->detLayer(irh->first->geographicalId())
			      << ", r/phi/z = "
			      << irh->first->globalPosition().perp() << " "
			      << irh->first->globalPosition().phi() << " "
			      << irh->first->globalPosition().z()
			      << std::endl;
      }
      
      std::sort( OverlapHits.begin(),OverlapHits.end(),SortHitTrajectoryPairsByGlobalPosition());
      if (debug_) std::cout<<"Sorted OverlapHits has size " <<OverlapHits.size() << std::endl;
      
      float workingBestChi2 = 1000000.0;
      std::vector<TrajectoryMeasurement> workingBestHits;
      
      std::vector<std::pair<TransientTrackingRecHit::ConstRecHitPointer,TrajectoryMeasurement*> >::iterator irh1;
      std::vector<std::pair<TransientTrackingRecHit::ConstRecHitPointer,TrajectoryMeasurement*> >::iterator irh2;
      for (irh1 =OverlapHits.begin(); irh1!=--OverlapHits.end(); ++irh1){
        theBestHits.clear();
        float running_chi2=0;
        if (debug_) std::cout << "Hit " << irh1-OverlapHits.begin()
			      << " on det " << irh1->first->det() 
			      << " detLayer " 
			      << theMeasurementTracker->geometricSearchTracker()->detLayer(irh1->first->geographicalId())
			      << ", r/phi/z = "
          
			      << irh1->first->globalPosition().perp() << " "
			      << irh1->first->globalPosition().phi() << " "
			      << irh1->first->globalPosition().z()
			      << std::endl;
        
        TrajectoryStateOnSurface currTsos = irh1->second->updatedState();
        TransientTrackingRecHit::ConstRecHitPointer rhit = irh1->first;
        theBestHits.push_back(*(irh1->second));
        if (debug_)  std::cout<<"Added first hit with chi2 = " << irh1->second->estimate() << std::endl;
        running_chi2 += irh1->second->estimate();
        for (irh2 = irh1; irh2!=OverlapHits.end(); ++irh2){
          if (irh2 == irh1) continue;
          TransientTrackingRecHit::ConstRecHitPointer rh = irh2->first;
          const GeomDet* det = irh2->first->det();
          // extrapolate the trajectory to the next hit
          TrajectoryStateOnSurface predTsos = thePropagator->propagate(currTsos, det->surface());
          // test if matches
          if (predTsos.isValid()){
            MeasurementEstimator::HitReturnType est = theEstimator->estimate(predTsos, *rh);
            if (debug_)  std::cout<<"Added overlap hit with est = " << est.first << "   " << est.second << std::endl;
            if (est.first){
              TrajectoryMeasurement tm(predTsos, currTsos, &(*rh),est.second,
                                       theMeasurementTracker->geometricSearchTracker()->detLayer(rh->geographicalId()));
              theBestHits.push_back(tm);
              running_chi2 += est.second ;
            }
            else { // couldn't add 2nd hit so return best single hit
            }
          }
          
        }
        if (theBestHits.size()==dtmmap.size()){ // added the best hit in every layer
          if (debug_) std::cout<<"Added all "<<theBestHits.size()<<" hits out of " << dtmmap.size() << std::endl;
          break;
        }
        // Didn't add hits from every Det
        if (theBestHits.size() < dtmmap.size()){
          if (debug_) std::cout<<"Added only "<<theBestHits.size()<<" hits out of " << dtmmap.size() << std::endl;
          // Take the combination with the most hits
          if (theBestHits.size() > workingBestHits.size()){
            if (debug_) std::cout<<"Current combo has more hits so replace best" << std::endl;
            workingBestHits = theBestHits;
          }
          // has same number of hits as best, so check chi2
          else if (theBestHits.size() == workingBestHits.size()){ 
            if (running_chi2< workingBestChi2){
              if (debug_) std::cout<<"Current combo has same # of hits but lower chi2 so replace best" << std::endl;
              workingBestHits = theBestHits;
              workingBestChi2 = running_chi2;
            }
          }
        }
      }
      if (theBestHits.size()<2){
        if (debug_) std::cout<<"Only one good hit in overlap"<<std::endl;
        if (debug_) std::cout<<" Added hit on layer on det " 
			     << theBestTM.recHit()->det() 
			     << " detLayer " 
			     << theMeasurementTracker->geometricSearchTracker()->detLayer(theBestTM.recHit()->geographicalId())
			     << ", r/phi/z = "
			     << theBestTM.recHit()->globalPosition().perp() << " "
			     << theBestTM.recHit()->globalPosition().phi() << " "
			     << theBestTM.recHit()->globalPosition().z()
			     << " with estimate " << theBestTM.estimate()<<std::endl ;
        theBestHits.clear();
        theBestHits.push_back(theBestTM);
      }
      
    }
    else {
      if (debug_)std::cout << "ERROR: Unexpected size from DTMMAP = " << dtmmap.size() << std::endl;
      theBestHits.push_back(theBestTM);
    }
  }
  
  return theBestHits;
}


//bool RoadSearchTrackCandidateMakerAlgorithm::chooseStartingLayers( RoadSearchCloud::RecHitVector& recHits, int ilayer0,
bool RoadSearchTrackCandidateMakerAlgorithm::chooseStartingLayers( std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >& recHitsByLayer,
								   //int ilayer0,
								   std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilyr0,
								   const std::multimap<int, const DetLayer*>& layer_map,
								   std::set<const DetLayer*>& good_layers,
								   std::vector<const DetLayer*>& middle_layers ,
								   RoadSearchCloud::RecHitVector& recHits_middle)
{
      const unsigned int max_middle_layers = 2;

      // consider the best nFoundMin layers + other layers with only one hit      
      // This has implications, based on the way we locate the hits.  
      // For now, use only the low occupancy layers in the first pass
      //const int nfound_min = min_chunk_length-1;
      //const int nfound_min = 4;
      std::multimap<int, const DetLayer*>::const_iterator ilm = layer_map.begin();
      int ngoodlayers = 0;
      while (ilm != layer_map.end()) {
        if (ngoodlayers >= nFoundMin_ && ilm->first > 1) break;
        //if (ilm->first > 1) break;
        good_layers.insert(ilm->second);
        ++ngoodlayers;
        ++ilm;
      }

      // choose intermediate layers
      for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilayer = ilyr0+1;
	   ilayer != recHitsByLayer.end(); ++ilayer) {
        // only use useful layers
        if (good_layers.find(ilayer->first) == good_layers.end()) continue;
        // only use stereo layers
        if (!CosmicReco_ && !lstereo[ilayer-recHitsByLayer.begin()]) continue;
        middle_layers.push_back(ilayer->first);
        if (middle_layers.size() >= max_middle_layers) break;
      }
      
      for (std::vector<const DetLayer*>::iterator ml = middle_layers.begin();
	   ml!=middle_layers.end();++ml){
	unsigned int middle_layers_found = 0;
	for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilyr = recHitsByLayer.begin();
	     ilyr != recHitsByLayer.end(); ++ilyr) {
	  if (ilyr->first == *ml){
	    for (RoadSearchCloud::RecHitVector::const_iterator ih = ilyr->second.begin();
		 ih != ilyr->second.end(); ++ih) {
	      recHits_middle.push_back(*ih);
	    }
	    ++middle_layers_found;
	  }
	  if (middle_layers_found == middle_layers.size()) continue;
	}

      }
      return (recHits_middle.size()>0);
}

FreeTrajectoryState RoadSearchTrackCandidateMakerAlgorithm::initialTrajectory(const edm::EventSetup& es,
									      const TrackingRecHit* theInnerHit,
									      const TrackingRecHit* theOuterHit)
{
  FreeTrajectoryState fts;

          GlobalPoint inner = trackerGeom->idToDet(theInnerHit->geographicalId())->surface().toGlobal(theInnerHit->localPosition());
          GlobalPoint outer = trackerGeom->idToDet(theOuterHit->geographicalId())->surface().toGlobal(theOuterHit->localPosition());
          
          LogDebug("RoadSearch") << "inner hit: r/phi/z = "<< inner.perp() << " " << inner.phi() << " " << inner.z() ;
          LogDebug("RoadSearch") << "outer hit: r/phi/z = "<< outer.perp() << " " << outer.phi() << " " << outer.z() ;
          
          // hits should be reasonably separated in r
          const double dRmin = 0.1; // cm
          if (outer.perp() - inner.perp() < dRmin) return fts;
          //GlobalPoint vertexPos(0,0,0);
	  const double dr2 = initialVertexErrorXY_*initialVertexErrorXY_;
          //const double dr2 = 0.0015*0.0015;
          //const double dr2 = 0.2*0.2;
          const double dz2 = 5.3*5.3;

	  // linear z extrapolation of two hits have to be inside tracker ( |z| < 275 cm)
	  FastLine linearFit(outer, inner);
	  double z_0 = -linearFit.c()/linearFit.n2();
	  if ( std::abs(z_0) > 275 && !CosmicReco_ ) return fts;

          GlobalError vertexErr(dr2,
                                0, dr2,
                                0, 0, dz2);
          //TrivialVertex vtx( vertexPos, vertexErr);
          //FastHelix helix(outerHit.globalPosition(),
          //              (*innerHit).globalPosition(),
          //              vtx.position());
          
          double x0=0.0,y0=0.0,z0=0.0;
	  double phi0 = -999.0;
          if (NoFieldCosmic_){
            phi0=atan2(outer.y()-inner.y(),outer.x()-inner.x());
	    thePropagator = theAloPropagator;//GC
	    if (inner.y()<outer.y()){
	      if (debug_) std::cout<<"Flipping direction!!!" << std::endl;
	      thePropagator = theRevPropagator;
	      phi0=phi0-M_PI;
	    } 
            double alpha=atan2(inner.y(),inner.x());
            double d1=sqrt(inner.x()*inner.x()+inner.y()*inner.y());
            double d0=-d1*sin(alpha-phi0); x0=d0*sin(phi0); y0=-d0*cos(phi0);
            double l1=0.0,l2=0.0;
            if (fabs(cos(phi0))>0.1){
              l1=(inner.x()-x0)/cos(phi0);l2=(outer.x()-x0)/cos(phi0);
            }else{
              l1=(inner.y()-y0)/sin(phi0);l2=(outer.y()-y0)/sin(phi0);
            }
            z0=(l2*inner.z()-l1*outer.z())/(l2-l1);
          }
          //FastHelix helix(outer, inner, vertexPos, es);
          FastHelix helix(outer, inner, GlobalPoint(x0,y0,z0), es);
          if (!NoFieldCosmic_ && !helix.isValid()) return fts;
          
          AlgebraicSymMatrix55 C = AlgebraicMatrixID();
          float zErr = vertexErr.czz();
          float transverseErr = vertexErr.cxx(); // assume equal cxx cyy
          C(3, 3) = transverseErr;
          C(4, 4) = zErr;
          CurvilinearTrajectoryError initialError(C);

          if (NoFieldCosmic_) {
	    TrackCharge q = 1;	    
	    FastLine flfit(outer, inner);
	    double dzdr = -flfit.n1()/flfit.n2();
	    if (inner.y()<outer.y()) dzdr*=-1;

	    GlobalPoint XYZ0(x0,y0,z0);
	    if (debug_) std::cout<<"Initial Point (x0/y0/z0): " << x0 <<'\t'<< y0 <<'\t'<< z0 << std::endl;
	    GlobalVector PXYZ(cosmicSeedPt_*cos(phi0),
			      cosmicSeedPt_*sin(phi0),
			      cosmicSeedPt_*dzdr);
	    GlobalTrajectoryParameters thePars(XYZ0,PXYZ,q,magField);
	    AlgebraicSymMatrix66 CErr = AlgebraicMatrixID();
	    CErr *= 5.0;
	    // CErr(3,3) = (theInnerHit->localPositionError().yy()*theInnerHit->localPositionError().yy() +
	    //		 theOuterHit->localPositionError().yy()*theOuterHit->localPositionError().yy());
	    fts = FreeTrajectoryState(thePars,
				      CartesianTrajectoryError(CErr));
	    if (debug_) std::cout<<"\nInitial CError (dx/dy/dz): " << CErr(1,1) <<'\t'<< CErr(2,2) <<'\t'<< CErr(3,3) << std::endl;
	    if (debug_) std::cout<<"\n\ninner dy = " << theInnerHit->localPositionError().yy() <<"\t\touter dy = " << theOuterHit->localPositionError().yy() << std::endl;
	  }
	  else {
	    fts = FreeTrajectoryState( helix.stateAtVertex().parameters(), initialError);
	  }
	  //                       RoadSearchSeedFinderAlgorithm::initialError( *outerHit, *(*innerHit),
          //                                  vertexPos, vertexErr));

	  return fts;
}

FreeTrajectoryState RoadSearchTrackCandidateMakerAlgorithm::initialTrajectoryFromTriplet(const edm::EventSetup& es,
											 const TrackingRecHit* theInnerHit,
											 const TrackingRecHit* theMiddleHit,
											 const TrackingRecHit* theOuterHit)
{
  FreeTrajectoryState fts;

          GlobalPoint inner = trackerGeom->idToDet(theInnerHit->geographicalId())->surface().toGlobal(theInnerHit->localPosition());
          GlobalPoint middle= trackerGeom->idToDet(theMiddleHit->geographicalId())->surface().toGlobal(theMiddleHit->localPosition());
          GlobalPoint outer = trackerGeom->idToDet(theOuterHit->geographicalId())->surface().toGlobal(theOuterHit->localPosition());
          
          LogDebug("RoadSearch") << "inner hit: r/phi/z = "<< inner.perp() << " " << inner.phi() << " " << inner.z() ;
          LogDebug("RoadSearch") << "middlehit: r/phi/z = "<< inner.perp() << " " << inner.phi() << " " << inner.z() ;
          LogDebug("RoadSearch") << "outer hit: r/phi/z = "<< outer.perp() << " " << outer.phi() << " " << outer.z() ;
          
          // hits should be reasonably separated in r
          const double dRmin = 0.1; // cm
          if (outer.perp() - inner.perp() < dRmin) return fts;
	  const double dr2 = initialVertexErrorXY_*initialVertexErrorXY_;
          const double dz2 = 5.3*5.3;

	  // linear z extrapolation of two hits have to be inside tracker ( |z| < 275 cm)
	  FastLine linearFit(outer, inner);
	  double z_0 = -linearFit.c()/linearFit.n2();
	  if ( std::abs(z_0) > 275 && !CosmicReco_ ) return fts;


          FastHelix helix(outer, middle, inner, es);
	  // check that helix is OK
          if (!helix.isValid() || 
	      std::isnan(helix.stateAtVertex().parameters().momentum().perp())) return fts;
	  // simple cuts on pt and dz
	  if (helix.stateAtVertex().parameters().momentum().perp() < 0.5 ||
	      std::abs(helix.stateAtVertex().parameters().position().z()) > 550.0)
	    return fts;

          AlgebraicSymMatrix55 C = AlgebraicMatrixID();
          float zErr = dz2;
          float transverseErr = dr2; // assume equal cxx cyy
          C(3, 3) = transverseErr;
          C(4, 4) = zErr;
          CurvilinearTrajectoryError initialError(C);


	  thePropagator = theAloPropagator;//GC
	  GlobalVector gv=helix.stateAtVertex().parameters().momentum();
	  float charge=helix.stateAtVertex().parameters().charge();

	  if (CosmicReco_ && gv.y()>0){
	    if (debug_) std::cout<<"Flipping direction!!!" << std::endl;
	    thePropagator = theRevPropagator;
	    gv=-1.*gv;
	    charge=-1.*charge;
	  }

	  GlobalTrajectoryParameters Gtp(inner,gv,int(charge),&(*magField));
	  fts = FreeTrajectoryState(Gtp, initialError);

	 //fts = FreeTrajectoryState( helix.stateAtVertex().parameters(), initialError);

	  return fts;
}



Trajectory RoadSearchTrackCandidateMakerAlgorithm::createSeedTrajectory(FreeTrajectoryState& fts,
									const TrackingRecHit* theInnerHit,
									const DetLayer* theInnerHitLayer)

{
  Trajectory theSeedTrajectory;

  // Need to put the first hit on the trajectory
  const GeomDet* innerDet = trackerGeom->idToDet((theInnerHit)->geographicalId());
  const TrajectoryStateOnSurface innerState = 
    thePropagator->propagate(fts,innerDet->surface());
  if ( !innerState.isValid() ||
       std::isnan(innerState.globalMomentum().perp())) {
    if (debug_) std::cout<<"*******DISASTER ********* seed doesn't make it to first hit!!!!!" << std::endl;
    return theSeedTrajectory;
  }
  TransientTrackingRecHit::RecHitPointer intrhit = ttrhBuilder->build(theInnerHit);
  // if this first hit is a matched hit, it should be updated for the trajectory
  const SiStripMatchedRecHit2D *origHit = dynamic_cast<const SiStripMatchedRecHit2D *>(theInnerHit);
  if (origHit !=0){
    const GluedGeomDet *gdet = dynamic_cast<const GluedGeomDet*>(innerDet);
    const SiStripMatchedRecHit2D *corrHit = theHitMatcher->match(origHit,gdet,innerState.localDirection());
    if (corrHit!=0){
      intrhit = ttrhBuilder->build(&(*corrHit));
      delete corrHit;
    }
  }
  
  MeasurementEstimator::HitReturnType est = theEstimator->estimate(innerState, *intrhit);
  if (!est.first) return theSeedTrajectory;	    
   TrajectoryStateOnSurface innerUpdated= theUpdator->update( innerState,*intrhit);                         
   if (debug_) std::cout<<"InnerUpdated: " << innerUpdated << std::endl;
  if (!innerUpdated.isValid() ||
       std::isnan(innerUpdated.globalMomentum().perp())) {
    if (debug_) std::cout<<"Trajectory updated with first hit is invalid!!!" << std::endl;
    return theSeedTrajectory;
  }
  TrajectoryMeasurement tm = TrajectoryMeasurement(innerState, innerUpdated, &(*intrhit),est.second,theInnerHitLayer);
  
  PTrajectoryStateOnDet pFirstStateTwo = trajectoryStateTransform::persistentState(innerUpdated,
									  intrhit->geographicalId().rawId());
  edm::OwnVector<TrackingRecHit> newHitsTwo;
  newHitsTwo.push_back(intrhit->hit()->clone());
  
  TrajectorySeed tmpseedTwo = TrajectorySeed(pFirstStateTwo, 
					     newHitsTwo,
					     alongMomentum);
  if (thePropagator->propagationDirection()==oppositeToMomentum) {
    tmpseedTwo = TrajectorySeed(pFirstStateTwo, 
				newHitsTwo,
				oppositeToMomentum);
  }

  
  //Trajectory seedTraj(tmpseedTwo, alongMomentum);
  theSeedTrajectory = Trajectory(tmpseedTwo, tmpseedTwo.direction());
  
  theSeedTrajectory.push(tm,est.second);
  
  return theSeedTrajectory;
}



std::vector<Trajectory> RoadSearchTrackCandidateMakerAlgorithm::extrapolateTrajectory(const Trajectory& theTrajectory,
										      RoadSearchCloud::RecHitVector& theLayerHits,
										      const DetLayer* innerHitLayer,
										      const TrackingRecHit* outerHit,
										      const DetLayer* outerHitLayer)
{
  std::vector<Trajectory> newTrajectories;

              for(RoadSearchCloud::RecHitVector::const_iterator ihit = theLayerHits.begin();
                  ihit != theLayerHits.end(); ihit++) {
		Trajectory traj = theTrajectory;
                const DetLayer* thisLayer =
		  theMeasurementTracker->geometricSearchTracker()->detLayer((*ihit)->geographicalId());
                if (thisLayer == innerHitLayer){
                  // Right now we are assuming that ONLY single hit layers are in this initial collection
                  //if (thisLayer == innerHitLayer && !(ihit->recHit() == innerHit->recHit())){
                  //  if (debug_) std::cout<<"On inner hit layer, but have wrong hit?!?!?" << std::endl;
                  continue;
		}
                if (thisLayer == outerHitLayer){
		  LocalPoint p1 = (*ihit)->localPosition();
		  LocalPoint p2 = outerHit->localPosition();
		  if (p1.x()!=p2.x() || p1.y()!=p2.y()) continue;
		}
                // extrapolate
                
                TransientTrackingRecHit::RecHitPointer rhit = ttrhBuilder->build(*ihit);
                
                if (debug_){
                  if (rhit->isValid()) {
                    LogDebug("RoadSearch") << "RecHit " << ihit-theLayerHits.begin()
					   << ", det " << rhit->det() << ", r/phi/z = "
					   << rhit->globalPosition().perp() << " "
					   << rhit->globalPosition().phi() << " "
					   << rhit->globalPosition().z();
                  } else {
                    LogDebug("RoadSearch") << "RecHit " << ihit-theLayerHits.begin()
					   << " (invalid)";
                  }
                }
                
                const GeomDet* det = trackerGeom->idToDet(rhit->geographicalId());
                
                TrajectoryStateOnSurface predTsos;
                TrajectoryStateOnSurface currTsos;
                
                if (traj.measurements().empty()) {
                  //predTsos = thePropagator->propagate(fts, det->surface());
		  if (debug_) std::cout<<"BIG ERROR!!! How did we make it to here with no trajectory measurements?!?!?"<<std::endl;
                } else {
                  currTsos = traj.measurements().back().updatedState();
                  predTsos = thePropagator->propagate(currTsos, det->surface());
                }
                if (!predTsos.isValid()){
                  continue;
                }
		GlobalVector propagationDistance = predTsos.globalPosition() - currTsos.globalPosition();
		if (propagationDistance.mag() > maxPropagationDistance) continue;
		if (debug_) {
		  std::cout << "currTsos (x/y/z): " 
			    << currTsos.globalPosition().x() << " / "
			    << currTsos.globalPosition().y() << " / "
			    << currTsos.globalPosition().z() << std::endl;
		  std::cout << "predTsos (x/y/z): " 
			    << predTsos.globalPosition().x() << " / "
			    << predTsos.globalPosition().y() << " / "
			    << predTsos.globalPosition().z() << std::endl;
		  std::cout << "Propagation distance1 is " << propagationDistance.mag() << std::endl;
		}
                TrajectoryMeasurement tm;
                if (debug_){
                  std::cout<< "trajectory at r/z=" <<  det->surface().position().perp() 
			   << "  " <<  det->surface().position().z() 
			   << ", hit " << ihit-theLayerHits.begin()
			   << " local prediction " << predTsos.localPosition().x() 
			   << " +- " << sqrt(predTsos.localError().positionError().xx()) 
			   << ", hit at " << rhit->localPosition().x() << " +- " << sqrt(rhit->localPositionError().xx())
			   << std::endl;
                }
                
                // update
		// first correct for angle

		const SiStripMatchedRecHit2D *origHit = dynamic_cast<const SiStripMatchedRecHit2D *>(*ihit);
		if (origHit !=0){
		  const GluedGeomDet *gdet = dynamic_cast<const GluedGeomDet*>(rhit->det());
		  const SiStripMatchedRecHit2D *corrHit = theHitMatcher->match(origHit,gdet,predTsos.localDirection());
		  if (corrHit!=0){
		    rhit = ttrhBuilder->build(&(*corrHit));
		    delete corrHit;
		  }
		}

		MeasurementEstimator::HitReturnType est = theEstimator->estimate(predTsos, *rhit);
		if (debug_) std::cout << "estimation: " << est.first << " " << est.second << std::endl;
		if (!est.first) continue;
		currTsos = theUpdator->update(predTsos, *rhit);
		tm = TrajectoryMeasurement(predTsos, currTsos, &(*rhit),est.second,thisLayer);
		traj.push(tm,est.second);
		newTrajectories.push_back(traj);


	      }

	    return newTrajectories;
}


TrackCandidateCollection RoadSearchTrackCandidateMakerAlgorithm::PrepareTrackCandidates(std::vector<Trajectory>& theTrajectories)
{

  TrackCandidateCollection theCollection;

  theTrajectoryCleaner->clean(theTrajectories);
  
  //==========NEW CODE ==========
  
  if(CosmicTrackMerging_) {
    
    //generate vector of *valid* trajectories -> traj
    std::vector<Trajectory> traj;

    //keep track of trajectories which are used during merging
    std::vector<bool> trajUsed;

    for (std::vector<Trajectory>::iterator it = theTrajectories.begin(); it != theTrajectories.end(); ++it) {
      if (it->isValid()) {
	traj.push_back(*it);
	trajUsed.push_back(false);
      }
    }
    
    if(debugCosmics_) {
      std::cout << "==========ENTERING COSMIC MODE===========" << std::endl;
      //      int t=0;
      for (std::vector<Trajectory>::iterator it = traj.begin(); it != traj.end(); it++) {
	std::cout << "Trajectory " << it-traj.begin() << " has "<<it->recHits().size()<<" hits and is valid: " << it->isValid() << std::endl;
	TransientTrackingRecHit::ConstRecHitContainer itHits = it->recHits();
	

	for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator rhit=itHits.begin(); rhit!=itHits.end(); ++rhit)
	  std::cout << "-->good hit position: " << (*rhit)->globalPosition().x() << ", " 
		    << (*rhit)->globalPosition().y() << ", "<< (*rhit)->globalPosition().z() << std::endl;

      }
    }

    //double nested looop to find trajectories that match in phi
    for ( unsigned int i = 0; i < traj.size(); ++i) {
      if (trajUsed[i]) continue;    
      for ( unsigned int j = i+1; j != traj.size(); ++j) {
	if (trajUsed[j]) continue;    
	
	if (debugCosmics_) std::cout<< "Trajectory " <<i<< " has "<<traj[i].recHits().size()<<" hits with chi2=" << traj[i].chiSquared() << " and is valid"<<std::endl;
	if (debugCosmics_) std::cout<< "Trajectory " <<j<< " has "<<traj[j].recHits().size()<<" hits with chi2=" << traj[j].chiSquared() << " and is valid"<<std::endl;           
	
	TrajectoryMeasurement firstTraj1 = traj[i].firstMeasurement();
	TrajectoryMeasurement firstTraj2 = traj[j].firstMeasurement();
	TrajectoryStateOnSurface firstTraj1TSOS = firstTraj1.updatedState();
	TrajectoryStateOnSurface firstTraj2TSOS = firstTraj2.updatedState();

	
	if(debugCosmics_) 
	  std::cout << "phi1: " << firstTraj1TSOS.globalMomentum().phi() 
		    << " phi2: " << firstTraj2TSOS.globalMomentum().phi() 
		    << " --> delta_phi: " << firstTraj1TSOS.globalMomentum().phi()-firstTraj2TSOS.globalMomentum().phi() << std::endl;
	
	//generate new trajectory if delta_phi<0.3
	//use phi of momentum vector associated to *innermost* hit of trajectories
	if( fabs(firstTraj1TSOS.globalMomentum().phi()-firstTraj2TSOS.globalMomentum().phi())<0.3 ) {
	  if(debugCosmics_) std::cout << "-->match successful" << std::endl;
	} else {
	  if(debugCosmics_) std::cout << "-->match not successful" << std::endl;
	}
	if( fabs(firstTraj1TSOS.globalMomentum().phi()-firstTraj2TSOS.globalMomentum().phi())>0.3 ) continue;
	
	
	//choose starting trajectory: use trajectory in lower hemisphere (with y<0) to start new combined trajectory
	//use y position of outermost hit
	TrajectoryMeasurement lastTraj1 = traj[i].lastMeasurement();
	TrajectoryMeasurement lastTraj2 = traj[j].lastMeasurement();
	TrajectoryStateOnSurface lastTraj1TSOS = lastTraj1.updatedState();
	TrajectoryStateOnSurface lastTraj2TSOS = lastTraj2.updatedState();
	
	if(debugCosmics_){
	  std::cout<<"Traj1 first (x/y/z): " 
		   << firstTraj1TSOS.globalPosition().x() <<" / "
		   << firstTraj1TSOS.globalPosition().y() <<" / "
		   << firstTraj1TSOS.globalPosition().z() 
		   << "   phi: " << firstTraj1TSOS.globalMomentum().phi() << std::endl;
	  std::cout<<"Traj1  last (x/y/z): " 
		   << lastTraj1TSOS.globalPosition().x() <<" / "
		   << lastTraj1TSOS.globalPosition().y() <<" / "
		   << lastTraj1TSOS.globalPosition().z() 
		   << "   phi: " << lastTraj1TSOS.globalMomentum().phi() << std::endl;

	  std::cout<<"Traj2 first (x/y/z): " 
		   << firstTraj2TSOS.globalPosition().x() <<" / "
		   << firstTraj2TSOS.globalPosition().y() <<" / "
		   << firstTraj2TSOS.globalPosition().z()
		   << "   phi: " << firstTraj2TSOS.globalMomentum().phi() << std::endl;
	  std::cout<<"Traj2  last (x/y/z): " 
		   << lastTraj2TSOS.globalPosition().x() <<" / "
		   << lastTraj2TSOS.globalPosition().y() <<" / "
		   << lastTraj2TSOS.globalPosition().z()
		   << "   phi: " << lastTraj2TSOS.globalMomentum().phi() << std::endl;

	}

	Trajectory *upperTrajectory, *lowerTrajectory;
	
	TrajectoryStateOnSurface lowerTSOS1,upperTSOS1;
	if (lastTraj1TSOS.globalPosition().y()<firstTraj1TSOS.globalPosition().y()) {
	  lowerTSOS1=lastTraj1TSOS; upperTSOS1=firstTraj1TSOS;
	}
	else {
	  lowerTSOS1=firstTraj1TSOS; upperTSOS1=lastTraj1TSOS;
	}
	TrajectoryStateOnSurface lowerTSOS2;
	if (lastTraj2TSOS.globalPosition().y()<firstTraj2TSOS.globalPosition().y()) lowerTSOS2=lastTraj2TSOS;
	else lowerTSOS2=firstTraj2TSOS;
	if(lowerTSOS1.globalPosition().y() > lowerTSOS2.globalPosition().y()) {
	  if(debugCosmics_) 
	    std::cout << "-->case A: "<< lowerTSOS1.globalPosition().y() << " > " << lowerTSOS2.globalPosition().y() << std::endl;
	  
	  upperTrajectory = &(traj[i]);
	  lowerTrajectory = &(traj[j]);
	  
	} else {
	  if(debugCosmics_) 
	    std::cout << "-->case B: "<< lowerTSOS1.globalPosition().y() << " < " << lowerTSOS2.globalPosition().y() << std::endl;
	  
	  upperTrajectory = &(traj[j]);
	  lowerTrajectory = &(traj[i]);
	} 
	
	std::vector<Trajectory> freshStartUpperTrajectory = theSmoother->trajectories(*upperTrajectory);
	std::vector<Trajectory> freshStartLowerTrajectory = theSmoother->trajectories(*lowerTrajectory);
	//--JR
	if (freshStartUpperTrajectory.empty() || freshStartLowerTrajectory .empty()){
	  if (debugCosmics_) std::cout << " the smoother has failed."<<std::endl;
	  continue;
	}
	//--JR
	TrajectoryStateOnSurface NewFirstTsos = freshStartUpperTrajectory.begin()->lastMeasurement().updatedState();
	TrajectoryStateOnSurface forwardTsos  = freshStartUpperTrajectory.begin()->firstMeasurement().forwardPredictedState();
	TrajectoryStateOnSurface backwardTsos = freshStartUpperTrajectory.begin()->lastMeasurement().backwardPredictedState();

	Trajectory freshStartTrajectory = *freshStartUpperTrajectory.begin();
	if(debugCosmics_) std::cout << "seed hit updatedState: " << NewFirstTsos.globalMomentum().x() << ", " << NewFirstTsos.globalMomentum().y() << ", " << NewFirstTsos.globalMomentum().z()  <<  std::endl;
	if(debugCosmics_) std::cout << "seed hit updatedState (pos x/y/z): " << NewFirstTsos.globalPosition().x() << ", " << NewFirstTsos.globalPosition().y() << ", " << NewFirstTsos.globalPosition().z()  <<  std::endl;
	if(debugCosmics_) std::cout << "seed hit forwardPredictedState: " << forwardTsos.globalMomentum().x() << ", " << forwardTsos.globalMomentum().y() << ", " << forwardTsos.globalMomentum().z()  <<  std::endl;
	if(debugCosmics_) std::cout << "seed hit forwardPredictedState (pos x/y/z): " << forwardTsos.globalPosition().x() << ", " << forwardTsos.globalPosition().y() << ", " << forwardTsos.globalPosition().z()  <<  std::endl;
	if(debugCosmics_) std::cout << "seed hit backwardPredictedState: " << backwardTsos.globalMomentum().x() << ", " << backwardTsos.globalMomentum().y() << ", " << backwardTsos.globalMomentum().z() <<  std::endl;
	if(debugCosmics_) std::cout << "seed hit backwardPredictedState (pos x/y/z): " << backwardTsos.globalPosition().x() << ", " << backwardTsos.globalPosition().y() << ", " << backwardTsos.globalPosition().z() <<  std::endl;
	
	if(debugCosmics_) std::cout<<"#hits for upper trajectory: " << freshStartTrajectory.measurements().size() << std::endl;
	
	//loop over hits in upper trajectory and add them to lower trajectory
	TransientTrackingRecHit::ConstRecHitContainer ttHits = lowerTrajectory->recHits(splitMatchedHits_);
	
	if(debugCosmics_) std::cout << "loop over hits in lower trajectory..." << std::endl;
	
	bool addHitToFreshStartTrajectory = false;
	bool propagationFailed = false;
	int lostHits = 0;
	for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator rhit=ttHits.begin(); rhit!=ttHits.end(); ++rhit){

	  if(debugCosmics_ && lostHits>0){
	    std::cout << " Lost " << lostHits << " of " << ttHits.size() << " on lower trajectory " << std::endl;
	    std::cout << " Lost " << ((float)lostHits/(float)ttHits.size()) << " of hits of on lower trajectory " << std::endl;
	  }
	  if ((float)lostHits/(float)ttHits.size() > 0.5) {
	      propagationFailed = true;
	      break;
	  }
	  if(debugCosmics_) std::cout << "-->hit position: " << (*rhit)->globalPosition().x() << ", " << (*rhit)->globalPosition().y() << ", "<< (*rhit)->globalPosition().z() << std::endl;
	  
	  TrajectoryStateOnSurface predTsos;
	  TrajectoryStateOnSurface currTsos;
	  TrajectoryMeasurement tm;
	  
	  TransientTrackingRecHit::ConstRecHitPointer rh = (*rhit);



	  /*
	  if( rh->isValid() ) { 
	    if(debugCosmics_) std::cout << "-->hit is valid"<<std::endl;
	    const GeomDet* det = trackerGeom->idToDet(rh->geographicalId());
	    
	    std::vector<TrajectoryMeasurement> measL = freshStartTrajectory.measurements();
	    bool alreadyUsed = false;	      
	    for (std::vector<TrajectoryMeasurement>::iterator mh=measL.begin();mh!=measL.end();++mh) {
	      if (rh->geographicalId().rawId()==mh->recHit()->geographicalId().rawId()) {
		if (debugCosmics_) std::cout << "this hit is already in the trajectory, skip it" << std::endl;
		alreadyUsed = true;
		break;
	      }
	    }
	    if (alreadyUsed) continue;
	    //std::vector<TrajectoryMeasurement> measU = freshStartUpperTrajectory[0].measurements();
	    if (freshStartTrajectory.direction()==0) {
	      std::vector<TrajectoryMeasurement>::iterator ml;
	      for (ml=measL.begin();ml!=measL.end();++ml) {
		if (debugCosmics_)  std::cout << "hit y="<<ml->recHit()->globalPosition().y()
					      << " tsos validity fwd="<<ml->forwardPredictedState().isValid() 
					      << " bwd="<<ml->backwardPredictedState().isValid() 
					      << " upd="<<ml->updatedState().isValid() 
					      <<std::endl;
		if (ml->recHit()->globalPosition().y()>(*rhit)->globalPosition().y() && ml!=measL.begin()) {
		  break;
		}
	      }
	      if ((ml-1)->forwardPredictedState().isValid()) currTsos = (ml-1)->forwardPredictedState();
	      else currTsos = (ml-1)->backwardPredictedState();
	      
	      if (debugCosmics_) std::cout << "currTsos y=" << currTsos.globalPosition().y() << std::endl;
	    }
	    else {
	      std::vector<TrajectoryMeasurement>::reverse_iterator ml;
	      for (ml=measL.rbegin();ml!=measL.rend();++ml) {
		if (debugCosmics_) std::cout << "hit y="<<ml->recHit()->globalPosition().y()
					     << " tsos validity fwd="<<ml->forwardPredictedState().isValid() 
					     << " bwd="<<ml->backwardPredictedState().isValid() 
					     << " upd="<<ml->updatedState().isValid() 
					     <<std::endl;
		if (ml->recHit()->globalPosition().y()>(*rhit)->globalPosition().y() && ml!=measL.rbegin()) {
		  break;
		}
	      }
	      if ((ml-1)->forwardPredictedState().isValid()) {
		currTsos = (ml-1)->forwardPredictedState();
	      }
	      else {
		currTsos = (ml-1)->backwardPredictedState();
	      }
	      
	      if (debugCosmics_) std::cout << "reverse. currTsos y=" << currTsos.globalPosition().y() << std::endl;
	    }
	    
	    
	  }
	  */


	  if( rh->isValid() ) { 
	    if(debugCosmics_) std::cout << "-->hit is valid"<<std::endl;
	    const GeomDet* det = trackerGeom->idToDet(rh->geographicalId());
	    if( addHitToFreshStartTrajectory==false ) {
	      //first hit from upper trajectory that is being added to lower trajectory requires usage of backwardPredictedState (of lower trajectory)
	      currTsos = freshStartTrajectory.lastMeasurement().backwardPredictedState();
	    } else {
	      //remaining hits from upper trajectory that are being added to lower trajectory require usage of forwardPredictedState
	      currTsos = freshStartTrajectory.lastMeasurement().forwardPredictedState();
	    }
	    
	    if(debugCosmics_) std::cout << "current TSOS: " << currTsos.globalPosition().x() << ", " << currTsos.globalPosition().y() << ", " << currTsos.globalPosition().z() << ", " << std::endl;
	    
	    predTsos = theAloPropagator->propagate(currTsos, det->surface());
	    
	    if (!predTsos.isValid()) {
	      if(debugCosmics_) std::cout<<"predTsos is not valid!" <<std::endl;
	      //propagationFailed = true;
	      ++lostHits;
	      //break;
	      continue;
	    }
	    GlobalVector propagationDistance = predTsos.globalPosition() - currTsos.globalPosition();
	    if (propagationDistance.mag() > maxPropagationDistance) continue;
	    
	    if(debugCosmics_) std::cout << "predicted TSOS: " << predTsos.globalPosition().x() << ", " << predTsos.globalPosition().y() << ", " << predTsos.globalPosition().z() << ", " << std::endl;
	    MeasurementEstimator::HitReturnType est = theEstimator->estimate(predTsos, *rh);
	    if (!est.first) {
	      if(debugCosmics_) std::cout<<"Failed to add one of the original hits on a low occupancy layer!!!!" << std::endl;
	      //propagationFailed = true;
	      ++lostHits;
	      //break;
	      continue;
	    }
	    
	    currTsos = theUpdator->update(predTsos, *rh);
	    if(debugCosmics_) std::cout << "current updated TSOS: " << currTsos.globalPosition().x() << ", " << currTsos.globalPosition().y() << ", " << currTsos.globalPosition().z() << ", " << std::endl;
	    tm = TrajectoryMeasurement(predTsos, currTsos,&(*rh),est.second,theMeasurementTracker->geometricSearchTracker()->detLayer(rh->geographicalId()));
	    freshStartTrajectory.push(tm,est.second);
	    addHitToFreshStartTrajectory=true;
	    
	  }

	  if(debugCosmics_) std::cout<<"#hits for new trajectory (his from upper trajectory added): " << freshStartTrajectory.measurements().size() << std::endl;
	}

	if (propagationFailed) {
	  if (debugCosmics_) std::cout<<"Propagation failed so go to next trajectory" << std::endl;
	  continue;
	}

	//put final trajectory together
	if(debugCosmics_) std::cout << "put final trajectory together..." << std::endl;
	edm::OwnVector<TrackingRecHit> goodHits;
	TransientTrackingRecHit::ConstRecHitContainer tttempHits = freshStartTrajectory.recHits(splitMatchedHits_);
	
	for (int k=tttempHits.size()-1; k>=0; k--) {
	  if(debugCosmics_) std::cout << "-->good hit position: " << tttempHits[k]->globalPosition().x() << ", " << tttempHits[k]->globalPosition().y() << ", "<< tttempHits[k]->globalPosition().z() << std::endl;
	  goodHits.push_back(tttempHits[k]->hit()->clone());
	}
	TrajectoryStateOnSurface firstState;
	
	// check if Trajectory from seed is on first hit of the cloud, if not, propagate
	// exclude if first state on first hit is not valid
	DetId FirstHitId = (*(freshStartTrajectory.recHits().end()-1))->geographicalId(); //was begin
	
	TrajectoryMeasurement maxYMeasurement = freshStartTrajectory.lastMeasurement();
	const GeomDet* det = trackerGeom->idToDet(FirstHitId);
	firstState = theAnalyticalPropagator->propagate(maxYMeasurement.updatedState(),det->surface());
	if (firstState.isValid() == false) continue;    
	PTrajectoryStateOnDet const & state = trajectoryStateTransform::persistentState(firstState,FirstHitId.rawId());
	
	//generate new trajectory seed
	TrajectoryStateOnSurface firstTSOS = freshStartTrajectory.lastMeasurement().updatedState();
	if(debugCosmics_) std::cout << "generate new trajectory seed with hit (x/y/z): " << firstTSOS.globalPosition().x() << ", " << firstTSOS.globalPosition().y() << ", " << firstTSOS.globalPosition().z() << ", " << std::endl;
	TransientTrackingRecHit::ConstRecHitPointer rhit = freshStartTrajectory.lastMeasurement().recHit();
	PTrajectoryStateOnDet const & pFirstState = trajectoryStateTransform::persistentState(NewFirstTsos,rhit->geographicalId().rawId());
	edm::OwnVector<TrackingRecHit> newHits;
	newHits.push_back(rhit->hit()->clone());
	TrajectorySeed tmpseed = TrajectorySeed(pFirstState,newHits,alongMomentum);
	
	theCollection.push_back(TrackCandidate(goodHits,freshStartTrajectory.seed(),state));
	
	//trajectory usage
	trajUsed[i]=true;
	trajUsed[j]=true;
	
      } //for loop trajectory2
      
    } //for loop trajectory1

    //add all trajectories to the resulting vector if they have *not* been used by the trajectory merging algorithm
    for ( unsigned int i = 0; i < traj.size(); ++i) {
      
      if (trajUsed[i]==true) continue;

      if (debugCosmics_) std::cout<< "Trajectory (not merged) has "<<traj[i].recHits().size()<<" hits with chi2=" << traj[i].chiSquared() << " and is valid? "<< traj[i].isValid()<<std::endl;
      edm::OwnVector<TrackingRecHit> goodHits;
      TransientTrackingRecHit::ConstRecHitContainer ttHits = traj[i].recHits(splitMatchedHits_);
      for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator rhit=ttHits.begin(); rhit!=ttHits.end(); ++rhit){
	goodHits.push_back((*rhit)->hit()->clone());
      }
      TrajectoryStateOnSurface firstState;
      
      // check if Trajectory from seed is on first hit of the cloud, if not, propagate
      // exclude if first state on first hit is not valid
      DetId FirstHitId = (*(traj[i].recHits().begin()))->geographicalId();
      
      // propagate back to first hit    
      TrajectoryMeasurement firstMeasurement = traj[i].measurements()[0];
      const GeomDet* det = trackerGeom->idToDet(FirstHitId);
      firstState = theAnalyticalPropagator->propagate(firstMeasurement.updatedState(), det->surface());	  
      if (firstState.isValid()) {
	PTrajectoryStateOnDet state = trajectoryStateTransform::persistentState(firstState,FirstHitId.rawId());
	theCollection.push_back(TrackCandidate(goodHits,traj[i].seed(),state));
      }
    }
    if (debugCosmics_) std::cout << "Original collection had " << theTrajectories.size() 
				 << " candidates.  Merged collection has " << theCollection.size() << std::endl;
  } //if(CosmicTrackMerging_)
  
  
  if(!CosmicTrackMerging_) {
     for (std::vector<Trajectory>::const_iterator it = theTrajectories.begin(); it != theTrajectories.end(); it++) {
        if (debug_) std::cout<< " Trajectory has "<<it->recHits().size()<<" hits with chi2=" << it->chiSquared() << " and is valid? "<<it->isValid()<<std::endl;
        if (it->isValid()){

           edm::OwnVector<TrackingRecHit> goodHits;
           TransientTrackingRecHit::ConstRecHitContainer ttHits = it->recHits(splitMatchedHits_);
           for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator rhit=ttHits.begin(); rhit!=ttHits.end(); ++rhit){
              goodHits.push_back((*rhit)->hit()->clone());
           }
           TrajectoryStateOnSurface firstState;

           // check if Trajectory from seed is on first hit of the cloud, if not, propagate
           // exclude if first state on first hit is not valid
           DetId FirstHitId = (*(it->recHits().begin()))->geographicalId();

           // propagate back to first hit    
           TrajectoryMeasurement firstMeasurement = it->measurements()[0];
           const GeomDet* det = trackerGeom->idToDet(FirstHitId);
           firstState = theAnalyticalPropagator->propagate(firstMeasurement.updatedState(), det->surface());	  
           if (firstState.isValid() == false) continue;    
           PTrajectoryStateOnDet state = trajectoryStateTransform::persistentState(firstState,FirstHitId.rawId());
           theCollection.push_back(TrackCandidate(goodHits,it->seed(),state));
        }
     }
  }

  return theCollection;
}
