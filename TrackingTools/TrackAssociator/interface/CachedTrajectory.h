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
// $Id: CachedTrajectory.h,v 1.7 2007/06/27 07:09:11 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/MuonChamberMatch.h"
#include "TrackingTools/TrackAssociator/interface/FiducialVolume.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <deque>

class CachedTrajectory {
 public:
   CachedTrajectory();
   enum TrajectorType { IpToEcal, IpToHcal, IpToHO, FullTrajectory };

   void reset_trajectory();
   
   /// propagate through the whole detector, returns true if successful
   bool propagateAll(const SteppingHelixStateInfo& initialState);
   
   void propagateForward(SteppingHelixStateInfo& state, float distance);

   /// get fast to a given DetId surface using cached trajectory
   TrajectoryStateOnSurface propagate(const Plane* plane);
 
   /// calculate trajectory change (Theta,Phi)
   /// delta = final - original
   std::pair<float,float> trajectoryDelta( TrajectorType );
   
   void setPropagator(const Propagator* ptr){	propagator_ = ptr; }
   void setStateAtIP(const SteppingHelixStateInfo& state){ stateAtIP_ = state; }
   
   /// get a set of points representing the trajectory between two cylinders
   /// of radius R1 and R2 and length L1 and L2. Parameter steps defines
   /// maximal number of steps in the detector.
   void getTrajectory(std::vector<SteppingHelixStateInfo>&,
		      const FiducialVolume&,
		      int steps = 4);
   
   void findEcalTrajectory(const FiducialVolume&);
   void findHcalTrajectory(const FiducialVolume&);
   void findHOTrajectory(const FiducialVolume&);

   const std::vector<SteppingHelixStateInfo>& getEcalTrajectory();
   const std::vector<SteppingHelixStateInfo>& getHcalTrajectory();
   const std::vector<SteppingHelixStateInfo>& getHOTrajectory();

   SteppingHelixStateInfo getStateAtEcal();
   SteppingHelixStateInfo getStateAtHcal();
   SteppingHelixStateInfo getStateAtHO();
   
   //get the innermost state of the whole trajectory 
   SteppingHelixStateInfo getInnerState();
   //get the outermost state of the whole trajectory 
   SteppingHelixStateInfo getOuterState();
   
   // specify the detector global boundaries to limit the propagator
   // units: cm
   // HINT: use lower bounds to limit propagateAll() action within
   //       smaller region, such as ECAL for example
   void setMaxDetectorRadius(float r = 800.){ maxRho_ = r;}
   void setMaxDetectorLength(float l = 2200.){ maxZ_ = l/2.;}
   void setMaxHORadius(float r = 800.) { HOmaxRho_ = r;}
   void setMaxHOLength(float l = 2200.) { HOmaxZ_ = l/2.;}
   void setMinDetectorRadius(float r = 0.){ minRho_ = r;}
   void setMinDetectorLength(float l = 0.){ minZ_ = l/2.;}

   void setPropagationStep(float s = 20.){ step_ = s;}
   
 protected:
   
   static int sign (float number){
      if (number ==0) return 0;
      if (number > 0)
	return 1;
      else
	return -1;
   }
   
   std::pair<float,float> delta( const double& theta1,
				 const double& theta2,
				 const double& phi1,
				 const double& phi2);
   
   float distance(const Plane* plane, int index) {
      if (index<0 || fullTrajectory_.empty() || uint(index) >= fullTrajectory_.size()) return 0;
      return plane->localZ(fullTrajectory_[index].position());
   }
   
   std::deque<SteppingHelixStateInfo> fullTrajectory_;
   std::vector<SteppingHelixStateInfo> ecalTrajectory_;
   std::vector<SteppingHelixStateInfo> hcalTrajectory_;
   std::vector<SteppingHelixStateInfo> hoTrajectory_;
   SteppingHelixStateInfo stateAtIP_;
   
   bool fullTrajectoryFilled_;
   
   const Propagator* propagator_;
   
   float maxRho_;
   float maxZ_;
   float HOmaxRho_;
   float HOmaxZ_;
   float minRho_;
   float minZ_;
   float step_;

};
#endif
