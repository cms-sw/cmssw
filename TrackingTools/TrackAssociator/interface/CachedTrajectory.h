#ifndef TrackAssociator_CachedTrajectory_h
#define TrackAssociator_CachedTrajectory_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      CachedTrajectory
// 
/*

 Description: CachedTrajectory is a transient class, which stores a set of
 * trajectory states that can be used as starting points when there is need
 * propagate the same track to a number of different surfaces, which might 
 * not be know in advance.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: CachedTrajectory.h,v 1.1 2007/01/21 15:30:35 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/MuonChamberMatch.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class CachedTrajectory {
 public:
   CachedTrajectory();

   void reset_trajectory();
   
   // propagate through the whole detector, returns true if successful
   bool propagateAll(const SteppingHelixStateInfo& initialState);
   
   void propagateForward(SteppingHelixStateInfo& state, float distance);

   // get fast to a given DetId surface using cached trajectory
   TrajectoryStateOnSurface propagate(const Plane* plane);
 
   // calculate trajectory size 
   float trajectoryDeltaEta();
   float trajectoryDeltaPhi();
   void setPropagator(Propagator* ptr){	propagator_ = ptr; }
   
   // get a set of points representing the trajectory between two cylinders
   // of radius R1 and R2 and length L1 and L2. Step < 0 corresonds to the 
   // default step used during trajectory caching.

   void getTrajectory(std::vector<SteppingHelixStateInfo>&,
		      const float r1,
		      const float r2, 
		      const float l1, 
		      const float l2, 
		      const float step = -999.);
   
   const std::vector<SteppingHelixStateInfo>& getEcalTrajectory();
   const std::vector<SteppingHelixStateInfo>& getHcalTrajectory();
   const std::vector<SteppingHelixStateInfo>& getHOTrajectory();
   
   SteppingHelixStateInfo getStateAtEcal();
   SteppingHelixStateInfo getStateAtHcal();
   SteppingHelixStateInfo getStateAtHO();
   
   // specify the detector global boundaries to limit the propagator
   // units: cm
   // HINT: use lower bounds to limit propagateAll() action within
   //       smaller region, such as ECAL for example
   void setDetectorRadius(float r = 800.){ maxRho_ = r;}
   void setDetectorLength(float l = 2200.){ maxZ_ = l/2;}
   void setPropagationStep(float s = 20.){ step_ = s;}
   
 protected:
   
   static int sign (float number){
      if (number ==0) return 0;
      if (number == fabs(number))
	return 1;
      else
	return -1;
   }
   
   float distance(const Plane* plane, int index) {
      if (index<0 || fullTrajectory_.empty() || uint(index) >= fullTrajectory_.size()) return 0;
      return plane->localZ(fullTrajectory_[index].position());
   }
   
   std::vector<SteppingHelixStateInfo> fullTrajectory_;
   std::vector<SteppingHelixStateInfo> ecalTrajectory_;
   std::vector<SteppingHelixStateInfo> hcalTrajectory_;
   std::vector<SteppingHelixStateInfo> hoTrajectory_;
   
   bool fullTrajectoryFilled_;
   bool ecalTrajectoryFilled_;
   bool hcalTrajectoryFilled_;
   bool hoTrajectoryFilled_;
   
   Propagator* propagator_;
   
   float maxRho_;
   float maxZ_;
   float step_;

};
#endif
