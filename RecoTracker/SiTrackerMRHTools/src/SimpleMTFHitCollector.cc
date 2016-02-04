#include "RecoTracker/SiTrackerMRHTools/interface/SimpleMTFHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdatorMTF.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TSiTrackerMultiRecHit.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrajectoryMeasurement.h" //added
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include <vector>
#include <map>

#define _debug_SimpleMTFHitCollector_ 

using namespace std;

vector<TrajectoryMeasurement> 
SimpleMTFHitCollector::recHits(const std::map<int, vector<TrajectoryMeasurement> >& tmmap, 
			       int i,
			       double annealing) const{
  
  
  //it assumes that the measurements are sorted in the smoothing direction
  map< int, vector<TrajectoryMeasurement> >::const_iterator itmeas = tmmap.find(i);
  LogDebug("SimpleMTFHitCollector") << "found the element "<< i << "in the map" << std::endl;
  
  vector<TrajectoryMeasurement> meas = itmeas->second;
  LogDebug("SimpleMTFHitCollector") << "got the vector corresponding to " << i << " element of the map "<< std::endl;
  
  if (meas.empty()) 	
    return vector<TrajectoryMeasurement>();
  
  
  //TransientTrackingRecHit::ConstRecHitContainer result;
  vector<TrajectoryMeasurement> result;
  TrajectoryStateCombiner statecombiner;

  //loop on the vector<TM> from the bottom to the top (assumes the measurement are sorted in the smoothing direction)
  for(vector<TrajectoryMeasurement>::reverse_iterator imeas = meas.rbegin(); imeas != meas.rend(); imeas++) 
    
    {
      //check if the rechit is a valid one: if it is build a MultiRecHit
      if(imeas->recHit()->geographicalId().rawId())
	{
	  //define the rechit
	  TransientTrackingRecHit::ConstRecHitPointer rechit = imeas->recHit();
	  //const DetLayer* layer = imeas->layer();
	  
	  uint32_t id = imeas->recHit()->geographicalId().rawId();
	  
	  LogDebug("SimpleMTFHitCollector") << "Detector Id: " << id << std::endl;
	  
	  
	  std::vector<std::pair<int, TrajectoryMeasurement> > layermeas;
	  getMeasurements(layermeas, tmmap,*imeas,i);
	  
	  
	  //create a MTM
	  MultiTrajectoryMeasurement mtm = getTSOS(layermeas, rechit, i);
	  
	  //then build a MRH from a vector<TM> and a MTM...the buildMRH method has to be a dedicated one 	        	
	  TransientTrackingRecHit::ConstRecHitContainer hits;
	  
	  for (vector<std::pair<int, TrajectoryMeasurement> >::const_iterator ittmeas = layermeas.begin(); ittmeas != layermeas.end(); ittmeas++)
	    {
	      //TransientTrackingRecHit::ConstRecHitContainer sharedhits;

	      if(ittmeas->second.recHit()->isValid())
		{
		  
		  LogDebug("SimpleMTFHitCollector") << "this rechit in the vector layermeas is valid." << std::endl;
		  
		  int k=0;
		  //if the rechit is valid we push back the hit  
		  //hits.push_back(ittmeas->second.recHit());
		  
		  if (!hits.size())
		    {
		      hits.push_back(ittmeas->second.recHit());
		      
		      
		      //check if the rechits in hits are the same, and in case remove the hit from the vector
		    }

		  else
		    {
		      for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihits = hits.begin(); ihits != hits.end(); ihits++ )
			{
			  
			  
			  //we check if the rechits are the same and if the vector hits is not made of 1 hit,
			  //if it is made of more than 1 hit we pop back the hit 
			  if((*ihits)->hit()->sharesInput(ittmeas->second.recHit()->hit(), TrackingRecHit::all))
			    {
			      LogDebug("SimpleMTFHitCollector") << "the rechit coming from layermeas is the same as the first rechit "
							      << "we skip this rechit in building the multirechit" << std::endl;
			      // sharedhits.push_back(ittmeas->second.recHit());
			      
			      k++;
			      break;
			    }
			  ////
			}
		  
		  //if(sharedhits.size()) continue;
	      
		  //else
	      // {
	      //      hits.push_back(ittmeas->second.recHit()); 
	      //     sharedhits.clear();
	      //  }
	      
		      if(k==0)
			{
			  LogTrace("SimpleMTFHitCollector") << "This hit is valid ";
			  hits.push_back(ittmeas->second.recHit());
			}
		      
		    }
		  
		}

	      else continue;
	      
	    }
	  
	      
	  if(hits.size()==0)
	    {
	      result.push_back(*imeas);
	      LogDebug("SimpleMTFHitCollector") << "we found no valid rechits, so we fill the vector with the initial measurement" << std::endl;
	    }
	      
	  else
	    {
	      
	      //	      TrajectoryStateOnSurface state = layermeas.front().second.predictedState();
	      TrajectoryStateOnSurface state= statecombiner.combine(layermeas.front().second.predictedState(), layermeas.front().second.backwardPredictedState());

	      LogDebug("SimpleMTFHitCollector") << "we build a trajectory measurement from a vector of hits of size" << hits.size() <<std::endl;
	      
	      result.push_back(TrajectoryMeasurement(state,theUpdator->buildMultiRecHit(state, hits, &mtm, annealing)));
	    }
	  
	}

      
      
      //if the rechit is not valid, the measurement is pushed back as it is
      else 
	{
	  result.push_back(*imeas);
	}
      
    }
  
      
  LogTrace("MultiRecHitCollector") << "Original Measurement size "  
				   << meas.size() << " SimpleMTFHitCollector returned " 
				   << result.size() << " rechits";
  //results are sorted in the fitting direction
  return result;	
  
}


void
SimpleMTFHitCollector::getMeasurements(std::vector<std::pair<int, TrajectoryMeasurement> >& layermeas,
				       const std::map<int, vector<TrajectoryMeasurement> >& tmmap, 
				       TrajectoryMeasurement& pmeas,
				       int i) const{
  
  uint32_t id = pmeas.recHit()->geographicalId().rawId();
  
  LogDebug("SimpleMTFHitCollector") << "Detector Id: " << id << std::endl;
  
  //vector<std::pair<int, TrajectoryMeasurement> > layermeas;
  
  if( pmeas.recHit()->geographicalId().rawId() )
    {
      //search for hits only on the same detector, comparing the detId. Fill also the vector layermeas with the original measurement. 
      
      
      //use the method fastmeas to search for compatible hits and put it into a vector
      // vector<TrajectoryMeasurement> veclayermeas = 
      //	getMeasurementTracker()->idToDet
      //	(pmeas.recHit()->geographicalId().rawId())->fastMeasurements(pmeas.updatedState(), 
      //    pmeas.updatedState(), 
      //	*thePropagator, 
      //	*(getEstimator()));
  
  
  //LogDebug("SimpleMTFHitCollector") << " method veclayermeas returned a vector of size: " 
  //				<< veclayermeas.size() 
  //				<< std::endl;
      
      //        for(vector<TrajectoryMeasurement>::iterator iveclayermeas=veclayermeas.begin();
      //      iveclayermeas!=veclayermeas.end();
      //      iveclayermeas++)
      
      // {
      
	//  layermeas.push_back(make_pair(i,pmeas));
	  
      for (std::map< int, vector<TrajectoryMeasurement> >::const_iterator immap=tmmap.begin(); 
	   immap!=tmmap.end(); 
	   immap++)
	{
	  int ntraj2 = immap->first;
	  //LogDebug("SimpleMTFHitCollector") << " number of the trajectory examined: " << ntraj2 << std::endl;
	      //map< int, vector<TrajectoryMeasurement> >::const_iterator k = immap->second;
	  const vector<TrajectoryMeasurement> & vecmeas = immap->second;
	  
	  //    if(ntraj2 == i) continue;
	  
	  
	  for(vector<TrajectoryMeasurement>::const_reverse_iterator itvmeas = vecmeas.rbegin(); itvmeas!=vecmeas.rend(); itvmeas++)
	    {
	      //    if ( ( itvmeas->recHit()->geographicalId().rawId() == id) && (itvmeas->recHit()->isValid()) && !(itvmeas->recHit()->hit()->sharesInput(pmeas.recHit()->hit(), TrackingRecHit::all)) ) //modoficare per inludere il layer nella ricerca
	      
	      //if(itvmeas->recHit()->hit()->sharesInput(iveclayermeas->recHit()->hit(), TrackingRecHit::some) )
	      if ( (itvmeas->recHit()->geographicalId().rawId() == id) )
		{ 
		      LogDebug("SimpleMTFHitCollector") << "found a matching rechit in the collector";
		      //    if(iveclayermeas->updatedState() == itvmeas->updatedState())
		      
		      layermeas.push_back(make_pair(ntraj2,*itvmeas));
		      
		}
	      
	      //if(itvmeas->recHit()->geographicalId().rawId() == id)
	      
	      //layermeas.push_back(make_pair(ntraj2,*itvmeas));
	      
	    }
	  
	}
	  
       
      //vector<TrajectoryMeasurement> 
      
      //currentLayerMeas=getMeasurementTracker()->idToDet(imeas->recHit()->geographicalId().rawId())->fastMeasurements(imeas->updatedState(), 
      //													 imeas->updatedState(), 
      //													 *thePropagator, 
      //													 *(getEstimator()));
      LogDebug("SimpleMTFHitCollector") << " built a vector of Trajectory Measurements all on the same layer, of size: " 
					<< layermeas.size() 
					<< std::endl;
    }
  
  else layermeas.push_back(make_pair(i,pmeas));
  
  
}

MultiTrajectoryMeasurement SimpleMTFHitCollector::getTSOS(const vector<std::pair<int, TrajectoryMeasurement> >& layermeas, 
							  TransientTrackingRecHit::ConstRecHitPointer rechit, 
							  int i) const{
  
  //we should search even for compatible tsos on this layer and build the multirechit with this knowledge...we can build an MTM.
  uint32_t id = rechit->geographicalId().rawId();
  
  LogDebug("SimpleMTFHitCollector") << "Detector Id: " << id << std::endl;

  LogDebug("SimpleMTFHitCollector") << "LayerMeas size: " << layermeas.size() << std::endl;
  
  const DetLayer* layer = layermeas.front().second.layer();
  
  std::map<int,TSOS> predictions; 
  std::map<int,TSOS> updates;
  LogDebug("SimpleMTFHitCollector") << " about to build a map with predicted and updated states " << std::endl;
  
  //we insert initially at least the original predicted & updated states
  //if(imeas->predictedState().isValid())	
  //  predictions[i] = imeas->predictedState();
  //else if ( imeas->backwardPredictedState().isValid() )
  //  predictions[i] = imeas->backwardPredictedState();
  //else viva il dale
  //  LogDebug("SimpleMTFHitCollector") << "error:invalid predicted and backward predicted states" << std::endl;
  
  //first fill the map with the state relative to the measurement we are analizing
  //if(imeas->updatedState().isValid())
  //  updates[i] = imeas->updatedState();
  
  //LogDebug("SimpleMTFHitCollector") << "Local Position of predicted state" << imeas->predictedState().localPosition() << std::endl;
  //LogDebug("SimpleMTFHitCollector") << "Local Position of updated state" << imeas->updatedState().localPosition() << std::endl;	    
  TrajectoryStateCombiner statecombiner;
  
  for (std::vector<std::pair<int,TrajectoryMeasurement> >::const_iterator itmeas=layermeas.begin(); itmeas!=layermeas.end(); itmeas++)
    {
      
      //we now have to build 2 maps, with the (predicted & updated) tsos for each track; then we can construct a MTM
      LogDebug("SimpleMTFHitCollector") << " number of the trajectory examined: " << itmeas->first << std::endl;
      
      //get the vector from the map
      //LogDebug("SimpleMTFHitCollector") << " size of the vector examining: " << vmeas.size() << std::endl;
      
      //if(ntraj == i)
      //{
      //	  LogDebug("SimpleMTFHitCollector") << " skipping trajectory number: " << ntraj << std::endl;
      //  continue;
      //	}
      
      //begin a cicle to search for measurements compatible(same layer)
      //check if the _detid_ of the original measurement is the same of this one and if the rechits are not the same one
      if ( (itmeas->second.recHit()->geographicalId().rawId() == id) ) //controlla che siano sullo stesso layer (cambiare!!!)
	{
	  
	  //	  LogDebug("SimpleMTFHitCollector") << "found a compatible hit " << std::endl;

	  //add an element to the maps with predicted and updated states 
	  if(itmeas->second.predictedState().isValid())
	    {
	      predictions[itmeas->first] = itmeas->second.predictedState();
	      //	      LogDebug("SimpleMTFHitCollector") << "predicted state inserted in the map " << std::endl;
	      if ( itmeas->second.backwardPredictedState().isValid() ){
		updates[itmeas->first]= statecombiner.combine(itmeas->second.predictedState(), itmeas->second.backwardPredictedState());
	      }
	    }
	  
	  //	  else if ( itmeas->second.backwardPredictedState().isValid() )
	  //	    {
	  //	      predictions[itmeas->first] = itmeas->second.backwardPredictedState();
	  //	      LogDebug("SimpleMTFHitCollector") << "bacwardpredicted state inserted in the map " << std::endl;
	  //	    }
	  
	  else 
	    LogDebug("SimpleMTFHitCollector") << "error:invalid predicted state" << std::endl; 
	  
// 	  if(itmeas->second.updatedState().isValid()){
	    
// 	    updates[itmeas->first] = itmeas->second.updatedState();
// 	    LogDebug("SimpleMTFHitCollector") << "updated state inserted in the map " << std::endl;
// 	  }
// 	   else if ( itmeas->second.predictedState().isValid() )
// 	     {
// 	       LogDebug("SimpleMTFHitCollector") << "error: invalid updated state, taking the predicted one instead " << std::endl; 
// 	       updates[itmeas->first] = itmeas->second.predictedState();
// 	     }
	  
// 	  else 
// 	    LogDebug("SimpleMTFHitCollector") << "error:invalid updated and backward predicted states" << std::endl;
	  
	  
	  //get the iterator to the predicted & updated states of the maps, to print the TSOS predicted and updated 
	  //	  map<int,TSOS>::iterator ipred = predictions.find(itmeas->first); 
	  //map<int,TSOS>::iterator iupd = updates.find(itmeas->first); 
	  
	  //controlla che siano giusti gli stati!!!
	  //LogDebug("SimpleMTFHitCollector") << "Local Position of predicted state " << ipred->second.localPosition() << std::endl
	  //			    << " of trajectory number " << ipred->first <<"\n" << std::endl;
	  //LogDebug("SimpleMTFHitCollector") << "Local Position of updated state " << iupd->second.localPosition() << std::endl
	  //<< " of trajectory number " << iupd->first <<"\n" << std::endl;
	  
	}
    }
  
  //create a MTM
  MultiTrajectoryMeasurement mtm = MultiTrajectoryMeasurement(rechit,predictions,updates,layer);
  return mtm;
}



MultiTrajectoryMeasurement SimpleMTFHitCollector::TSOSfinder(const std::map<int, vector<TrajectoryMeasurement> >& tmmap, 
							     TrajectoryMeasurement& pmeas,
							     int i) const{
  
  
  //it assumes that the measurements are sorted in the smoothing direction
  //map< int, vector<TrajectoryMeasurement> >::const_iterator itmeas = tmmap.find(i);
  //LogDebug("SimpleMTFHitCollector") << "found the element "<< i << "in the map" << std::endl;
  
  //vector<TrajectoryMeasurement> meas = itmeas->second;
  //LogDebug("SimpleMTFHitCollector") << "got the vector corresponding to " << i << " element of the map "<< std::endl;
  
  //if (meas.empty()) 	
  // return vector<TrajectoryMeasurement>();
  
  
  //TransientTrackingRecHit::ConstRecHitContainer result;
  vector<TrajectoryMeasurement> result;
   
  
  //  if(pmeas.recHit()->isValid())
    
    
    
      //define the rechit
      TransientTrackingRecHit::ConstRecHitPointer rechit = pmeas.recHit();
      
      //      const DetLayer* layer = pmeas.layer();
      
      uint32_t id = pmeas.recHit()->geographicalId().rawId();
      
      LogDebug("SimpleMTFHitCollector") << "Detector Id: " << id << std::endl;

      //get all the measurement on the same layer, searching in the map
      vector<std::pair<int, TrajectoryMeasurement> > layermeas; 
      getMeasurements(layermeas, tmmap,pmeas,i);
      
      
      //create a MTM, by knowledge of the measurements on the same layer and the rechit...
      return getTSOS(layermeas, rechit, i);
    
      
    }
  

	 
	 //then build a MRH from a vector<TM> and a MTM...the buildMRH method has to be a dedicated one 	        	
	 //TransientTrackingRecHit::ConstRecHitContainer hits;
	 
	 //for (vector<std::pair<int, TrajectoryMeasurement> >::const_iterator ittmeas = layermeas.begin(); ittmeas != layermeas.end(); ittmeas++){
	 // if (ittmeas->second.recHit()->getType() != TrackingRecHit::missing) {
	 //  LogTrace("SimpleMTFHitCollector") << "This hit is valid ";
	 //  hits.push_back(ittmeas->second.recHit());
	 //}
	 //}
	 //TrajectoryStateOnSurface state = layermeas.front().second.predictedState();
	 
	 

	 
	 //obsolete method,  not used in track reconstruction or MRH building
void SimpleMTFHitCollector::buildMultiRecHits(const vector<std::pair<int, TrajectoryMeasurement> >& vmeas, 
					      MultiTrajectoryMeasurement* mtm, 
					      vector<TrajectoryMeasurement>& result,
					      double annealing) const {
  
  if (vmeas.empty()) {
    LogTrace("SimpleMTFHitCollector") << "the search for the measurement on the same layer returned an empty vector...we have got a problem... " ; 
    //should we do something?
    //result.push_back(InvalidTransientRecHit::build(0,TrackingRecHit::missing));
    return;
    } 
  
  
  //the state is computed taking the vmeas (all on the same surface) and their predicted state
  TrajectoryStateOnSurface state = vmeas.front().second.predictedState();
  LogTrace("SimpleMTFHitCollector") << "Position (local) of the first measurement state: " << state.localPosition();
  LogTrace("SimpleMTFHitCollector") << "Position (global) of the first measurement state: " << state.globalPosition();
  
  if (state.isValid()==false){
    LogTrace("MultiRecHitCollector") << "first state is invalid; skipping ";
    return;
  }
  
  //vector<const TrackingRecHit*> hits;
  TransientTrackingRecHit::ConstRecHitContainer hits;
  
  for (vector<std::pair<int, TrajectoryMeasurement> >::const_iterator ittmeas = vmeas.begin(); ittmeas != vmeas.end(); ittmeas++){
    if (ittmeas->second.recHit()->getType() != TrackingRecHit::missing) {
      LogTrace("SimpleMTFHitCollector") << "This hit is valid ";
      hits.push_back(ittmeas->second.recHit());
    }
  }
  
  if (hits.empty()){
    LogTrace("MultiTrackFilterHitCollector") << "No valid hits found ";
    return;
  }
  
  LogTrace("SimpleMTFHitCollector") << "The hits vector has size: " << hits.size() << "\n"
				    << "and has first component global position: " << hits.front()->globalPosition();


  //for(std::map<int,TSOS>::const_iterator imtm=mtm->filteredStates().begin(); imtm!=mtm->filteredStates().end(); imtm++)
  // {
  //  LogDebug("SimpleMTFHitCollector::BuildMultiRecHits") << "TSOS number " << imtm->first << "\n" 
  //						   << "TSOS position " << imtm->second.localPosition() << std::endl;
  // }
  
  //LogDebug("SimpleMTFHitCollector::BuildMultiRecHits") << "Map Size:" << mtm->filteredStates().size() << "\n";


  //the work of building a concrete MRH is done by theUpdator->buildMultiRecHit method (modified, to include an mtm...)
  result.push_back(TrajectoryMeasurement(state,theUpdator->buildMultiRecHit(state, hits, mtm, annealing)));	

  
  //result.push_back(TrajectoryMeasurement(state,theUpdator->update(hits, state)));	
  
}


