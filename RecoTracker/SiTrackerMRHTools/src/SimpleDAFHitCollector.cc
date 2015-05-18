#include "RecoTracker/SiTrackerMRHTools/interface/SimpleDAFHitCollector.h"
#include "RecoTracker/SiTrackerMRHTools/interface/SiTrackerMultiRecHitUpdator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

#ifdef EDM_ML_DEBUG
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#endif


#include <vector>
#include <map>

#define _debug_SimpleDAFHitCollector_ 

using namespace std;

vector<TrajectoryMeasurement> SimpleDAFHitCollector::recHits(const Trajectory& traj, const MeasurementTrackerEvent *theMTE) const{

  LogTrace("MultiRecHitCollector") << " Calling SimpleDAFHitCollector::recHits" << std::endl;

  //WARNING: At the moment the trajectories has the measurements 
  //with reversed sorting after the track smoothing
  const vector<TrajectoryMeasurement> meas = traj.measurements();
  unsigned int hitcounter = 1;

  if (meas.empty()) return vector<TrajectoryMeasurement>();

  LogTrace("MultiRecHitCollector") << "  Original measurements are:";
  Debug(meas);

  //groups hits on a sensor by sensor with same Id of previous TM
  //we have to sort the TrajectoryMeasurements in the opposite way in the fitting direction
  vector<TrajectoryMeasurement> result;
  for(vector<TrajectoryMeasurement>::const_reverse_iterator itrajmeas = meas.rbegin(); itrajmeas < meas.rend();
      itrajmeas++, hitcounter++) {

      DetId id = itrajmeas->recHit()->geographicalId();
      MeasurementDetWithData measDet = theMTE->idToDet(id);
      tracking::TempMeasurements tmps;
      
      std::vector<const TrackingRecHit*> hits;
      std::vector<std::unique_ptr<const TrackingRecHit>> hitsOwner;

      TrajectoryStateOnSurface smoothtsos = itrajmeas->updatedState();
      //the error is scaled in order to take more "compatible" hits
//      if( smoothtsos.isValid() ) smoothtsos.rescaleError(10);

      TrajectoryStateOnSurface predtsos_fwd = itrajmeas->predictedState();
      TrajectoryStateOnSurface predtsos_bwd = itrajmeas->backwardPredictedState();
      if( !predtsos_fwd.isValid() || !predtsos_bwd.isValid() ){
        LogTrace("MultiRecHitCollector") << "Something wrong! no valid TSOS found in current group ";
        continue;
      }

      TrajectoryStateCombiner combiner;
      TrajectoryStateOnSurface combtsos;
      if (hitcounter == meas.size()) combtsos = predtsos_fwd;
      else if (hitcounter == 1) combtsos = predtsos_bwd;
      else combtsos = combiner(predtsos_bwd, predtsos_fwd);

      //collected hits compatible with the itrajmeas
      if( measDet.measurements(smoothtsos, *(getEstimator()), tmps)){

        LogTrace("MultiRecHitCollector") << "  Found " << tmps.size() << " compatible measurements";

        for (std::size_t i=0; i!=tmps.size(); ++i){

          DetId idtemps = tmps.hits[i]->geographicalId();

          if( idtemps == id && tmps.hits[i]->hit()->isValid() ) {
            LogTrace("MultiRecHitCollector") << "  This is valid with position " << tmps.hits[i]->hit()->localPosition() << " and error " << tmps.hits[i]->hit()->localPositionError();

            TransientTrackingRecHit::RecHitPointer transient = 
						   theUpdator->getBuilder()->build(tmps.hits[i]->hit());
            TrackingRecHit::ConstRecHitPointer preciseHit = theHitCloner.makeShared(transient,combtsos);
	    auto righthit = rightdimension(*preciseHit);
            hitsOwner.push_back(std::move(righthit));
            hits.push_back(hitsOwner.back().get());
          }

        }

        //the error was scaled, now is scaled back (even if this is not tightly necessary)
//        if (smoothtsos.isValid()) smoothtsos.rescaleError(0.1);

        //I will keep the Invalid hit, IF this is not the first one       
        if (hits.empty()){
          LogTrace("MultiRecHitCollector") << " -> but no valid hits found in current group.";

          if( result.empty() ) continue;

          result.push_back(TrajectoryMeasurement(predtsos_fwd,
                                        std::make_shared<InvalidTrackingRecHit>(measDet.mdet().geomDet(), TrackingRecHit::missing)));
        } else {
          //measurements in groups are sorted with increating chi2
          //sort( *hits.begin(), *hits.end(), TrajMeasLessEstim());
	  if(!itrajmeas->recHit()->isValid()) 
	    LogTrace("MultiRecHitCollector") << "  -> " << hits.size() << " valid hits for this sensor. (IT WAS INVALID!!!)";
          else LogTrace("MultiRecHitCollector") << "  -> " << hits.size() << " valid hits for this sensor.";

          //building a MultiRecHit out of each sensor group
          result.push_back(TrajectoryMeasurement(predtsos_fwd,theUpdator->buildMultiRecHit(hits, combtsos, measDet)));
        }
      } else {
          LogTrace("MultiRecHitCollector") << "  No measurements found in current group.";
          //the error was scaled, now is scaled back (even if this is not tightly necessary)
//          if (smoothtsos.isValid()) smoothtsos.rescaleError(0.1);

          if( result.empty() ) continue;

          result.push_back(TrajectoryMeasurement(predtsos_fwd,
                                        std::make_shared<InvalidTrackingRecHit>(measDet.mdet().geomDet(), TrackingRecHit::missing)));

      }
      

  }
  LogTrace("MultiRecHitCollector") << " Ending SimpleDAFHitCollector::recHits >> " << result.size();

  //LogTrace("MultiRecHitCollector") << "  New measurements are:";
  //Debug(result);

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
#ifdef EDM_ML_DEBUG
  for(vector<TrajectoryMeasurement>::const_iterator itrajmeas = TM.begin(); itrajmeas < TM.end();
      itrajmeas++) {
    if (itrajmeas->recHit()->isValid()){

      LogTrace("MultiRecHitCollector") << "  Valid Hit with DetId " << itrajmeas->recHit()->geographicalId().rawId() << " and dim:" << itrajmeas->recHit()->dimension()
                      //<< " type " << typeid(itrajmeas->recHit()).name()
                        << " local position " << itrajmeas->recHit()->hit()->localPosition()
                        << " global position " << itrajmeas->recHit()->hit()->globalPosition()
                        << " and r " << itrajmeas->recHit()->hit()->globalPosition().perp() ;

      DetId hitId = itrajmeas->recHit()->geographicalId();

      if(hitId.det() == DetId::Tracker) {
        if (hitId.subdetId() == StripSubdetector::TIB )
          LogTrace("MultiRecHitCollector") << "  I am TIB " << TIBDetId(hitId).layer();
        else if (hitId.subdetId() == StripSubdetector::TOB )
          LogTrace("MultiRecHitCollector") << "  I am TOB " << TOBDetId(hitId).layer();
        else if (hitId.subdetId() == StripSubdetector::TEC )
          LogTrace("MultiRecHitCollector") << "  I am TEC " << TECDetId(hitId).wheel();
        else if (hitId.subdetId() == StripSubdetector::TID )
          LogTrace("MultiRecHitCollector") << "  I am TID " << TIDDetId(hitId).wheel();
        else if (hitId.subdetId() == (int) PixelSubdetector::PixelBarrel )
          LogTrace("MultiRecHitCollector") << "  I am PixBar " << PXBDetId(hitId).layer();
        else if (hitId.subdetId() == (int) PixelSubdetector::PixelEndcap )
          LogTrace("MultiRecHitCollector") << "  I am PixFwd " << PXFDetId(hitId).disk();
        else
          LogTrace("MultiRecHitCollector") << "  UNKNOWN TRACKER HIT TYPE ";
      }
      else if(hitId.det() == DetId::Muon) {
        if(hitId.subdetId() == MuonSubdetId::DT) 
          LogTrace("MultiRecHitCollector") << "  I am DT " << DTWireId(hitId);
        else if (hitId.subdetId() == MuonSubdetId::CSC )
          LogTrace("MultiRecHitCollector") << "  I am CSC " << CSCDetId(hitId);
        else if (hitId.subdetId() == MuonSubdetId::RPC )
          LogTrace("MultiRecHitCollector") << "  I am RPC " << RPCDetId(hitId);
        else
          LogTrace("MultiRecHitCollector") << "  UNKNOWN MUON HIT TYPE ";
      }
      else
        LogTrace("MultiRecHitCollector") << "  UNKNOWN HIT TYPE ";


      LogTrace("MultiRecHitCollector") << "  TSOS predicted_fwd " << itrajmeas->predictedState().localPosition() ;
      LogTrace("MultiRecHitCollector") << "  TSOS predicted_bwd " << itrajmeas->backwardPredictedState().localPosition() ;
      LogTrace("MultiRecHitCollector") << "  TSOS smoothtsos " << itrajmeas->updatedState().localPosition() ;
    } else {
      LogTrace("MultiRecHitCollector") << "  Invalid Hit with DetId " << itrajmeas->recHit()->geographicalId().rawId();
    }
      LogTrace("MultiRecHitCollector") << "\n";
  }
#endif
}
