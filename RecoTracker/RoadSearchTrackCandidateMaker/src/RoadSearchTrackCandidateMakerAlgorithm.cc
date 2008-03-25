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
// $Author: noeding $
// $Date: 2007/10/10 18:30:40 $
// $Revision: 1.44 $
//

#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/RoadSearchTrackCandidateMaker/interface/RoadSearchTrackCandidateMakerAlgorithm.h"

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
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
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
#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchSeedFinderAlgorithm.h"
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
  theTransformer = new TrajectoryStateTransform;
  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits;
  
  NoFieldCosmic_  = conf_.getParameter<bool>("StraightLineNoBeamSpotCloud");
  MinChunkLength_ = conf_.getParameter<int>("MinimumChunkLength");
  nFoundMin_      = conf_.getParameter<int>("nFoundMin");
  
  measurementTrackerName_ = conf_.getParameter<std::string>("MeasurementTrackerName");
  
  debug_ = false;

}

RoadSearchTrackCandidateMakerAlgorithm::~RoadSearchTrackCandidateMakerAlgorithm() {
  delete theEstimator;
  delete theUpdator;
  delete theTransformer;
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
  
  theMeasurementTracker->update(e);
  //const MeasurementTracker*  theMeasurementTracker = new MeasurementTracker(es,mt_params); // will need this later
  
  thePropagator = new PropagatorWithMaterial(alongMomentum,.1057,&(*magField)); 
  theRevPropagator = new PropagatorWithMaterial(oppositeToMomentum,.1057,&(*magField)); 
  theAnalyticalPropagator = new AnalyticalPropagator(magField,anyDirection);

  KFTrajectorySmoother theSmoother(*theRevPropagator, *theUpdator, *theEstimator);
  
  // get hit matcher
  theHitMatcher = new SiStripRecHitMatcher(3.0);

  debug_ = false;
  //if (input->size()>0) debug_ = true;

  LogDebug("RoadSearch") << "Clean Clouds input size: " << input->size();
  if (debug_) std::cout << std::endl << std::endl
			<< "*** NEW EVENT: Clean Clouds input size: " << input->size() << std::endl;
  
  int i_c = 0;
  int nchit = 0;
  for ( RoadSearchCloudCollection::const_iterator cloud = input->begin(); cloud != input->end(); ++cloud ) {
    
    // fill rechits from cloud into new
    RoadSearchCloud::RecHitVector recHits = cloud->recHits();
    nchit = recHits.size();
    
    std::vector<Trajectory> CloudTrajectories;
    
    if (!NoFieldCosmic_){
      std::sort(recHits.begin(),recHits.end(),SortHitPointersByGlobalPosition(tracker.product(),alongMomentum));
    }
    else {
      std::sort(recHits.begin(),recHits.end(),SortHitPointersByY(*tracker));
    }

    const int nlost_max = 2;
	    
    // make a list of layers in cloud and mark stereo layers
    
    const int max_layers = 128;
    /*
    const DetLayer* layers[max_layers];
    bool lstereo[max_layers];
    int nhits_l[max_layers];
    int nlayers = 0;
    */

    nlayers = 0;

    //std::map<const DetLayer*, int> cloud_layer_reference; // for debugging
    std::multimap<const DetLayer*, const TrackingRecHit* > cloud_layer_map;
    std::map<const DetLayer*, int> cloud_layer_reference0; // for debugging
    std::multimap<const DetLayer*, const TrackingRecHit* >::const_iterator hiter;
    for (RoadSearchCloud::RecHitVector::const_iterator ih=recHits.begin(); ih!=recHits.end(); ++ih) {
      const DetLayer* hitLayer =
	theMeasurementTracker->geometricSearchTracker()->detLayer((*ih)->geographicalId());
      int ilayer = -1;
      hiter = cloud_layer_map.find(hitLayer);
      if (ih == recHits.begin() || hitLayer != layers[nlayers-1]) {
        
        if (hiter == cloud_layer_map.end()) {// new layer
          if (nlayers >= max_layers) break; // should never happen
          layers[nlayers] = hitLayer;
          lstereo[nlayers] = false;
          nhits_l[nlayers] = 0;
          cloud_layer_reference0.insert(std::make_pair(layers[nlayers], nlayers));
          ilayer = nlayers;
          ++nlayers;
        }
        else{
	  std::map<const DetLayer*, int>::const_iterator ilyr = cloud_layer_reference0.find(hitLayer);
          ilayer = ilyr->second;
        }
      }
      else {
        ilayer = nlayers-1;
      }
      ++nhits_l[ilayer];
      if ((*ih)->localPositionError().yy()<1.) lstereo[ilayer] = true;      
      cloud_layer_map.insert(std::make_pair(hitLayer, *ih));
      if (debug_) {
	GlobalPoint gp = trackerGeom->idToDet((*ih)->geographicalId())->surface().toGlobal((*ih)->localPosition());
	std::cout << "Hit "<< ih-recHits.begin()
		  << " r/z = " << gp.perp() << " " << gp.z()
		  <<" in layer " << hitLayer << " layer number " << ilayer
		  << " with " << nhits_l[ilayer] << "  hits " << std::endl;
      }
    }
    if (debug_) std::cout<<"CLOUD LAYER MAP SIZE IS " << cloud_layer_map.size() << std::endl;
    LogDebug("RoadSearch")<<"Cloud #"<<i_c<<" has "<<recHits.size()<<" hits in "<<cloud_layer_map.size()<<" layers ";
    
    
    // collect hits in cloud by layer
    std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > > RecHitsByLayer;
    std::map<const DetLayer*, int> cloud_layer_reference; // for debugging
    for(RoadSearchCloud::RecHitVector::const_iterator ihit = recHits.begin();
	ihit != recHits.end(); ihit++) {
      // only use useful layers
      const DetLayer* thisLayer =
	theMeasurementTracker->geometricSearchTracker()->detLayer((*ihit)->geographicalId());

      std::map<const DetLayer*, int>::const_iterator ilyr = cloud_layer_reference.find(thisLayer);
      if (ilyr==cloud_layer_reference.end())
	cloud_layer_reference.insert(std::make_pair( thisLayer, RecHitsByLayer.size()));
      if (!RecHitsByLayer.empty() && RecHitsByLayer.back().first == thisLayer) { // Old Layer
	RecHitsByLayer.back().second.push_back(*ihit);
      }
      else {
	if (NoFieldCosmic_) {
	  if (ilyr != cloud_layer_reference.end()){// Not a New Layer
	    int ilayer = ilyr->second;
	    (RecHitsByLayer.begin()+ilayer)->second.push_back(*ihit);
	  }
	  else{// New Layer
	    RoadSearchCloud::RecHitVector rhc;
	    rhc.push_back(*ihit);
	    RecHitsByLayer.push_back(std::make_pair(thisLayer, rhc));
	  }
	}
	else{ // Assume it is a new layer
	  RoadSearchCloud::RecHitVector rhc;
	  rhc.push_back(*ihit);
	  RecHitsByLayer.push_back(std::make_pair(thisLayer, rhc));
	}

      }
    }

    if (debug_){
      int ntothit = 0;
      for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilhv = RecHitsByLayer.begin();
	   ilhv != RecHitsByLayer.end(); ++ilhv) {
	RoadSearchCloud::RecHitVector theLayerHits = ilhv->second;
	for (RoadSearchCloud::RecHitVector::const_iterator ihit = theLayerHits.begin();
           ihit != theLayerHits.end(); ++ihit) {
	
	  GlobalPoint gp = trackerGeom->idToDet((*ihit)->geographicalId())->surface().toGlobal((*ihit)->localPosition());
	  std::cout << "Hit "<< ntothit
		    << " r/z = "
		    << gp.perp() << " " << gp.z()
		    <<" in layer " << ilhv->first 
		    << " is hit " << (ihit-theLayerHits.begin())+1 
		    << " of " << theLayerHits.size() << "  total hits " << std::endl;
	  ntothit++;
	}
      }
    }

    LogDebug("RoadSearch")<<"Cloud #"<<i_c<<" has "<<recHits.size()<<" hits in "<<RecHitsByLayer.size()<<" layers ";
    if (debug_) std::cout <<"Cloud "<<i_c<<" has "<<recHits.size()<<" hits in " <<RecHitsByLayer.size() << " layers ";
    ++i_c;

    if (debug_){
      std::cout<<"\n*** Test of New Data Structure:" << std::endl;
      for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilhv = RecHitsByLayer.begin();
	   ilhv != RecHitsByLayer.end(); ++ilhv) {
	std::cout<<"\t Layer " << ilhv-RecHitsByLayer.begin() << " has " << ilhv->second.size() << " hits " << std::endl;
      }
    }

    // try to start from all layers until the chunk is too short
    //
    
    for (std::vector<std::pair<const DetLayer*, RoadSearchCloud::RecHitVector > >::iterator ilyr0 = RecHitsByLayer.begin();
	 ilyr0 != RecHitsByLayer.end(); ++ilyr0) {

      uint ilayer0 = (uint)(ilyr0-RecHitsByLayer.begin());
      if (ilayer0 > RecHitsByLayer.size()-MinChunkLength_) continue;      

      std::vector<Trajectory> ChunkTrajectories;
      std::vector<Trajectory> CleanChunks;
      bool all_chunk_layers_used = false;
      
      if (debug_) std::cout  << "*** START NEW CHUNK --> layer range (" << ilayer0 << "-" << nlayers-1 << ")";

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
      for (int ilayer = ilayer0+1; ilayer < nlayers; ++ilayer) {
        layer_map.insert(std::make_pair(nhits_l[ilayer], layers[ilayer]));
        layer_reference.insert(std::make_pair(layers[ilayer], ilayer));
      }
      
      if (debug_) {
	std::cout<<std::endl<<"   Available layers are: " << std::endl;
	for (std::multimap<int, const DetLayer*>::iterator ilm1 = layer_map.begin();
	     ilm1 != layer_map.end(); ++ilm1) {
	  std::map<const DetLayer*, int>::iterator ilr = layer_reference.find(ilm1->second);
	  if (ilr != layer_reference.end() && debug_) 
	    std::cout << "Layer " << ilr->second << " with " << nhits_l[ilr->second]<<" hits" <<std::endl;;
	}
      }

      const int max_middle_layers = 2;
      std::set<const DetLayer*> the_good_layers;
      std::vector<const DetLayer*> the_middle_layers;
      RoadSearchCloud::RecHitVector the_recHits_middle;

      bool StartLayers = chooseStartingLayers(RecHitsByLayer,ilayer0,layer_map,the_good_layers,the_middle_layers,the_recHits_middle);
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
      int ngoodlayers = good_layers.size();

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

        for (RoadSearchCloud::RecHitVector::iterator outerHit = recHits_outer.begin();
             outerHit != recHits_outer.end(); ++outerHit) {
          
	  const DetLayer* outerHitLayer =
	    theMeasurementTracker->geometricSearchTracker()->detLayer((*outerHit)->geographicalId());

          if (debug_){
	    std::map<const DetLayer*, int>::iterator ilro = layer_reference.find(outerHitLayer);
	    if (ilro != layer_reference.end()) {
	      std::cout << "Try trajectory with Inner Hit on Layer " << ilayer0 << " and  " ;
	      std::cout << "Outer Hit on Layer " << ilro->second << std::endl;
	    }
	  }

	  FreeTrajectoryState fts = initialTrajectory(es,*innerHit,*outerHit);
	  if (!fts.hasError()) continue;

	  Trajectory seedTraj = createSeedTrajectory(fts,*innerHit,innerHitLayer);

          std::vector<Trajectory> rawTrajectories;          
          rawTrajectories.push_back(seedTraj);

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

	    if (debug_) std::cout<<"Used " << used_layers.size() << " layers out of " << ngoodlayers
				 << " good layers, so " << ngoodlayers - used_layers.size() << " missed "
				 << std::endl;
            if ((int)used_layers.size() < nFoundMin_) continue;
            int nlostlayers = ngoodlayers - used_layers.size();
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
          
          Trajectory temptraj = *i;
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
	    }
	  }
          
          // Loop over the layers in the cloud
          
	  std::set<const DetLayer*> final_layers;
          Trajectory::DataContainer::const_iterator im = tmv.begin();
          Trajectory::DataContainer::const_iterator im2 = tmv.begin();
          
          TrajectoryMeasurement firstMeasurement = i->firstMeasurement();
          const DetLayer* firstDetLayer = 
            theMeasurementTracker->geometricSearchTracker()->detLayer(firstMeasurement.recHit()->geographicalId());
          
          std::vector<Trajectory> freshStart = theSmoother.trajectories(*i);
          if (!freshStart.empty()){
            if (debug_) std::cout<< "Smoothing of trajectory " <<i-CleanChunks.begin() << " has succeeded with " 
				 <<freshStart.begin()->measurements().size() << " hits.  Now add hits." <<std::endl;
          }
          else {
            if (debug_) std::cout<< "Smoothing of trajectory " <<i-CleanChunks.begin() <<" has failed"<<std::endl;
            continue;
          }
          TrajectoryStateOnSurface NewFirstTsos = freshStart.begin()->lastMeasurement().updatedState();
          TransientTrackingRecHit::ConstRecHitPointer rh = freshStart.begin()->lastMeasurement().recHit();
          PTrajectoryStateOnDet* pFirstState = TrajectoryStateTransform().persistentState(NewFirstTsos,
                                                                                          rh->geographicalId().rawId());
          edm::OwnVector<TrackingRecHit> newHits;
          newHits.push_back(rh->hit()->clone());
          
          TrajectorySeed tmpseed = TrajectorySeed(*pFirstState, 
                                                  newHits,
                                                  alongMomentum);
          
          delete pFirstState;
          
          
          Trajectory newTrajectory(tmpseed,tmpseed.direction());
          
          const GeomDet* det = trackerGeom->idToDet(rh->geographicalId());
          TrajectoryStateOnSurface invalidState(new BasicSingleTrajectoryState(det->surface()));
          newTrajectory.push(TrajectoryMeasurement(invalidState, NewFirstTsos, rh, 0, firstDetLayer));
	  final_layers.insert(firstDetLayer);


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
	      std::cout<<"Trajectory has " << newTrajectory.measurements().size() << " with " << (RecHitsByLayer.end()-ilyr)
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
                MeasurementEstimator::HitReturnType est = theEstimator->estimate(predTsos, *rh);
                if (!est.first) {
                  if (debug_) std::cout<<"Failed to add one of the original hits on a low occupancy layer!!!!" << std::endl;
                  continue;
                }
                currTsos = theUpdator->update(predTsos, *rh);
                tm = TrajectoryMeasurement(predTsos, currTsos, &(*rh),est.second,ilyr->first);
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

  delete thePropagator;
  delete theRevPropagator; 
  delete theAnalyticalPropagator;
  delete theHitMatcher;
  if (debug_) std::cout<< "*** RS Found " << output.size() << " track candidates."<<std::endl;

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
      dmmap.insert(std::make_pair(*idet, predTsos));
    }
  }
  // evaluate hit residuals
  std::map<const GeomDet*, TrajectoryMeasurement> dtmmap;
  for (edm::OwnVector<TrackingRecHit>::const_iterator ih = theHits.begin();
       ih != theHits.end(); ++ih) {
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
								   int ilayer0,
								   const std::multimap<int, const DetLayer*>& layer_map,
								   std::set<const DetLayer*>& good_layers,
								   std::vector<const DetLayer*>& middle_layers ,
								   RoadSearchCloud::RecHitVector& recHits_middle)
{
      const uint max_middle_layers = 2;

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
        ++ngoodlayers;
        ++ilm;
      }

      for (std::multimap<int, const DetLayer*>::const_iterator ilm1 = layer_map.begin();
           ilm1 != ilm; ++ilm1) {
        good_layers.insert(ilm1->second);
      }
      
      // choose intermediate layers
      for (int ilayer = ilayer0+1; ilayer<nlayers; ++ilayer) {
        // only use useful layers
        if (good_layers.find(layers[ilayer]) == good_layers.end()) continue;
        // only use stereo layers
        if (!NoFieldCosmic_ && !lstereo[ilayer]) continue;
        middle_layers.push_back(layers[ilayer]);
        if (middle_layers.size() >= max_middle_layers) break;
      }
      
      for (std::vector<const DetLayer*>::iterator ml = middle_layers.begin();
	   ml!=middle_layers.end();++ml){
	int middle_layers_found = 0;
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
          const double dr2 = 0.0015*0.0015;
          //const double dr2 = 0.2*0.2;
          const double dz2 = 5.3*5.3;

	  // linear z extrapolation of two hits have to be inside tracker ( |z| < 275 cm)
	  FastLine linearFit(outer, inner);
	  double z_0 = -linearFit.c()/linearFit.n2();
	  if ( std::abs(z_0) > 275 ) return fts;

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
	    GlobalPoint XYZ0(x0,y0,z0);
	    GlobalVector PXYZ(cos(phi0),sin(phi0),dzdr);
	    GlobalTrajectoryParameters thePars(XYZ0,PXYZ,q,magField);
	    AlgebraicSymMatrix66 CErr = AlgebraicMatrixID();
	    fts = FreeTrajectoryState(thePars,
				      CartesianTrajectoryError(CErr));
	  }
	  else {
	    fts = FreeTrajectoryState( helix.stateAtVertex().parameters(), initialError);
	  }
	  //                       RoadSearchSeedFinderAlgorithm::initialError( *outerHit, *(*innerHit),
          //                                  vertexPos, vertexErr));

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
  if ( !innerState.isValid()) {
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
  TrajectoryMeasurement tm = TrajectoryMeasurement(innerState, innerUpdated, &(*intrhit),est.second,theInnerHitLayer);
  
  PTrajectoryStateOnDet* pFirstStateTwo = theTransformer->persistentState(innerUpdated,
									  intrhit->geographicalId().rawId());
  edm::OwnVector<TrackingRecHit> newHitsTwo;
  newHitsTwo.push_back(intrhit->hit()->clone());
  
  TrajectorySeed tmpseedTwo = TrajectorySeed(*pFirstStateTwo, 
					     newHitsTwo,
					     alongMomentum);
  delete pFirstStateTwo;
  
  //Trajectory seedTraj(tmpseedTwo, alongMomentum);
  theSeedTrajectory = Trajectory(tmpseedTwo, alongMomentum);
  
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
  for (std::vector<Trajectory>::const_iterator it = theTrajectories.begin();
       it != theTrajectories.end(); it++) {
    if (debug_) std::cout<< " Trajectory has "<<it->recHits().size()<<" hits with chi2="
			 <<it->chiSquared()<<" and is valid? "<<it->isValid()<<std::endl;
    if (it->isValid()){
    
      edm::OwnVector<TrackingRecHit> goodHits;
      TransientTrackingRecHit::ConstRecHitContainer ttHits = it->recHits();		
      for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator rhit=ttHits.begin(); 
	   rhit!=ttHits.end(); ++rhit){
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
      PTrajectoryStateOnDet *state = theTransformer->persistentState(firstState,FirstHitId.rawId());
      theCollection.push_back(TrackCandidate(goodHits,it->seed(),*state));
      delete state;

    }
  }

  return theCollection;
}
