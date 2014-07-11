#include "RecoTracker/SiTrackerMRHTools/interface/SimpleDAFHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"

#include <vector>
#include <map>

#define _debug_SimpleDAFHitCollector_ 

using namespace std;

vector<TrajectoryMeasurement> SimpleDAFHitCollector::recHits(const Trajectory& traj, const MeasurementTrackerEvent *theMTE) const{

  LogTrace("MultiRecHitCollector") << " Calling SimpleDAFHitCollector::recHits" << std::endl;

  //WARNING: At the moment the trajectories has the measurements 
  //with reversed sorting after the track smoothing
  const vector<TrajectoryMeasurement> meas = traj.measurements();

  if (meas.empty()) return vector<TrajectoryMeasurement>();

  LogTrace("MultiRecHitCollector") << "  Original measurements are:";
  Debug(meas);

  //groups hits on a sensor by sensor with same Id of previous TM
  //we have to sort the TrajectoryMeasurements in the opposite way in the fitting direction
  vector<TrajectoryMeasurement> result;
  for(vector<TrajectoryMeasurement>::const_reverse_iterator itrajmeas = meas.rbegin(); itrajmeas < meas.rend();
      itrajmeas++) {

      DetId id = itrajmeas->recHit()->geographicalId();
      MeasurementDetWithData measDet = theMTE->idToDet(id);
      tracking::TempMeasurements tmps;
      std::vector<const TrackingRecHit*> hits;

      TrajectoryStateOnSurface smoothState = itrajmeas->updatedState();
      //the error is scaled in order to take more "compatible" hits
      if (smoothState.isValid()) smoothState.rescaleError(10);

      TrajectoryStateOnSurface predState = itrajmeas->predictedState();
      if (!predState.isValid()){
        LogTrace("MultiRecHitCollector") << "Something wrong! no valid TSOS found in current group ";
        continue;
      }

      //collected hits compatible with the itrajmeas
      if( measDet.measurements(smoothState, *(getEstimator()), tmps)){

        LogTrace("MultiRecHitCollector") << "  Found " << tmps.size() << " compatible measurements";

        for (std::size_t i=0; i!=tmps.size(); ++i){

          DetId idtemps = tmps.hits[i]->geographicalId();

          if( idtemps == id && tmps.hits[i]->hit()->isValid() ) {

	    //fill with the right dimension hit
            if( itrajmeas->recHit()->dimension() == 1 ){
	      auto const & hit1 = tmps.hits[i]->hit();
	      auto const & thit = static_cast<BaseTrackerRecHit const&>(*hit1);
	      hits.push_back(clone(thit));
	    } else {
              hits.push_back(tmps.hits[i]->hit());
	    }

          }

        }

        //I will keep the Invalid hit, IF this is not the first one       
        if (hits.empty()){
          LogTrace("MultiRecHitCollector") << " -> but no valid hits found in current group.";

          if( result.empty() ) continue;

          result.push_back(TrajectoryMeasurement(predState,
                                        std::make_shared<InvalidTrackingRecHit>(measDet.mdet().geomDet(), TrackingRecHit::missing)));
        } else {
          //measurements in groups are sorted with increating chi2
          //sort( *hits.begin(), *hits.end(), TrajMeasLessEstim());

          LogTrace("MultiRecHitCollector") << " -> " << hits.size() << " valid hits for this sensor.";

          //building a MultiRecHit out of each sensor group
          result.push_back(TrajectoryMeasurement(predState,theUpdator->buildMultiRecHit(hits, smoothState)));
        }
      } else {
          LogTrace("MultiRecHitCollector") << "  No measurements found in current group.";

          if( result.empty() ) continue;

          result.push_back(TrajectoryMeasurement(predState,
                                        std::make_shared<InvalidTrackingRecHit>(measDet.mdet().geomDet(), TrackingRecHit::missing)));

      }
      
  }
  LogTrace("MultiRecHitCollector") << " Ending SimpleDAFHitCollector::recHits >> " << result.size();

  LogTrace("MultiRecHitCollector") << "  Original measurements are:";
  Debug(result);

  //adding a protection against too few hits and invalid hits 
  //(due to failed propagation on the same surface of the original hits)
  if (result.size()>2)
  {
    int hitcounter=0;
    //check if the vector result has more than 3 valid hits
    for (vector<TrajectoryMeasurement>::const_iterator iimeas = result.begin(); iimeas != result.end(); ++iimeas) {
      if(iimeas->recHit()->isValid()) hitcounter++;
    }

    if(hitcounter>2)
      return result;
    else return vector<TrajectoryMeasurement>();
  }

  else{return vector<TrajectoryMeasurement>();}

}


void SimpleDAFHitCollector::Debug( const std::vector<TrajectoryMeasurement> TM ) const
{
  for(vector<TrajectoryMeasurement>::const_iterator itrajmeas = TM.begin(); itrajmeas < TM.end();
      itrajmeas++) {
    if (itrajmeas->recHit()->isValid()){
      LogTrace("MultiRecHitCollector") << "  Valid Hit with DetId " << itrajmeas->recHit()->geographicalId().rawId() << " and dim:" << itrajmeas->recHit()->dimension()
                      //<< " type " << typeid(itrajmeas->recHit()).name()
                        << " local position " << itrajmeas->recHit()->hit()->localPosition()
                        << " global position " << itrajmeas->recHit()->hit()->globalPosition()
                        << " and r " << itrajmeas->recHit()->hit()->globalPosition().perp() ;
      LogTrace("MultiRecHitCollector") << "  TSOS predicted " << itrajmeas->predictedState().localPosition() ;
      LogTrace("MultiRecHitCollector") << "  TSOS smoothState " << itrajmeas->updatedState().localPosition() ;
            } else {
              LogTrace("MultiRecHitCollector") << "   Invalid Hit with DetId " << itrajmeas->recHit()->geographicalId().rawId();
            }
  }
}
