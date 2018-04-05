#include "RecoTracker/SiTrackerMRHTools/interface/MeasurementByLayerGrouper.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

vector<pair<const DetLayer*, vector<TrajectoryMeasurement> > > MeasurementByLayerGrouper::operator()(const vector<TM>& vtm) const{

	if(vtm.empty()) 
    		return vector<pair<const DetLayer*, vector<TM> > >();

  	vector<pair<const DetLayer*, vector<TM> > > result;
	result.reserve(vtm.size());

	vector<TM>::const_iterator start = vtm.begin();
	//here we assume that the TM on the same detLayer are consecutive (as it should)
  	while(start != vtm.end()) {
    		vector<TM>::const_iterator ipart = start;
		do {ipart++;}
    		while(ipart != vtm.end() && 
	  		getDetLayer(*start)==getDetLayer(*ipart) &&
			getDetLayer(*start) != nullptr  //the returned pointer will be 0 in case
                                                  //the measurement contains an invalid hit with no associated detid.
						  //This kind of hits are at most one per layer.
						  //this last condition avoids that 2 consecutive measurements of this kind
                                                  //are grouped in the same layer.
					          //it would be useful if invalid hit out of the active area were 
						  //given the detid reserved for the whole layer instead of 0    
			) ;
    
    		vector<TM> group(start, ipart);
    		result.push_back(pair<const DetLayer*, vector<TM> >(getDetLayer(*start),
							group));
    		start = ipart;
  	}
#ifdef debug_MeasurementByLayerGrouper_
	for (vector<pair<const DetLayer*, vector<TM> > >::const_iterator iter = result.begin(); iter != result.end(); iter++){
		LogTrace("MeasurementByLayerGrouper|SiTrackerMultiRecHitUpdator") << "DetLayer " << iter->first << " has " << iter->second.size() << " measurements"; 
	}
#endif
	
	
	return result;
}

const DetLayer* MeasurementByLayerGrouper::getDetLayer(const TM& tm) const {
	// if the DetLayer is set in the TM...  
	if (tm.layer()) return tm.layer();

	//if it corresponds to an invalid hit with no geomdet associated
        //we can't retrieve the  DetLayer
        //because unfortunately the detlayer is not set in these cases
	//returns 0 for the moment
	//to be revisited
	
        if (tm.recHit()->det()==nullptr){
	  LogDebug("MeasurementByLayerGrouper") <<"This hit has no geomdet associated skipping... ";
		return nullptr;
        }

	//now the cases in which the detid is set

	if (!theGeomSearch) {
		throw cms::Exception("MeasurementByLayerGrouper") << "Impossible to retrieve the det layer because it's not set in the TM and the pointer to the GeometricSearchTracker is 0 ";
		return nullptr;	
	}

	return theGeomSearch->detLayer(tm.recHit()->det()->geographicalId());
}
