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
// $Id: CachedTrajectory.h,v 1.22 2012/11/08 21:28:47 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TAMuonChamberMatch.h"
#include "TrackingTools/TrackAssociator/interface/FiducialVolume.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <deque>
#include "FWCore/Utilities/interface/Visibility.h"

std::vector<SteppingHelixStateInfo> 
propagateThoughFromIP(const SteppingHelixStateInfo& state,const Propagator* prop,
		      const FiducialVolume& volume,int nsteps,
		      float step, float minR, float minZ, float maxR, float maxZ);

class CachedTrajectory {
 public:

  const std::vector<SteppingHelixStateInfo>& getEcalTrajectory() const;
  const std::vector<SteppingHelixStateInfo>& getHcalTrajectory() const;
  const std::vector<SteppingHelixStateInfo>& getHOTrajectory() const;
  const std::vector<SteppingHelixStateInfo>& getPreshowerTrajectory() const;

private:
  friend class TrackDetectorAssociator;
  friend std::vector<SteppingHelixStateInfo> 
  propagateThoughFromIP(const SteppingHelixStateInfo& state,const Propagator* ptr,
			const FiducialVolume& volume,int nsteps,
			float step, float minR, float minZ, float maxR, float maxZ);

  CachedTrajectory();
  enum TrajectorType { IpToEcal, IpToHcal, IpToHO, FullTrajectory };
  enum WideTrajectoryType { Ecal, Hcal, HO };
  
  void reset_trajectory() dso_internal;
  
  /// propagate through the whole detector, returns true if successful
  bool propagateAll(const SteppingHelixStateInfo& initialState) dso_internal;
  
  void propagateForward(SteppingHelixStateInfo& state, float distance) dso_internal;
  void propagate(SteppingHelixStateInfo& state, const Plane& plane) dso_internal;
  void propagate(SteppingHelixStateInfo& state, const Cylinder& cylinder) dso_internal;
  
  /// get fast to a given DetId surface using cached trajectory
  TrajectoryStateOnSurface propagate(const Plane* plane) dso_internal;
  
  /// calculate trajectory change (Theta,Phi)
  /// delta = final - original
  std::pair<float,float> trajectoryDelta( TrajectorType ) dso_internal;
  
  void setPropagator(const Propagator* ptr) dso_internal { propagator_ = ptr; }
  void setStateAtIP(const SteppingHelixStateInfo& state) dso_internal { stateAtIP_ = state; }
  
  /// get a set of points representing the trajectory between two cylinders
  /// of radius R1 and R2 and length L1 and L2. Parameter steps defines
  /// maximal number of steps in the detector.
   void getTrajectory(std::vector<SteppingHelixStateInfo>&,
		      const FiducialVolume&,
		      int steps = 4) dso_internal;
  
  void findEcalTrajectory(const FiducialVolume&) dso_internal;
  void findHcalTrajectory(const FiducialVolume&) dso_internal;
  void findHOTrajectory(const FiducialVolume&) dso_internal;
  void findPreshowerTrajectory(const FiducialVolume&) dso_internal;
  
  std::vector<GlobalPoint>* getWideTrajectory(const std::vector<SteppingHelixStateInfo>&,
					      WideTrajectoryType) dso_internal;
  
  SteppingHelixStateInfo getStateAtEcal() dso_internal;
  SteppingHelixStateInfo getStateAtPreshower() dso_internal;
  SteppingHelixStateInfo getStateAtHcal() dso_internal;
  SteppingHelixStateInfo getStateAtHO() dso_internal;
  
  //get the innermost state of the whole trajectory 
  SteppingHelixStateInfo getInnerState() dso_internal;
  //get the outermost state of the whole trajectory 
  SteppingHelixStateInfo getOuterState() dso_internal;
  
  // specify the detector global boundaries to limit the propagator
   // units: cm
   // HINT: use lower bounds to limit propagateAll() action within
   //       smaller region, such as ECAL for example
  void setMaxDetectorRadius(float r = 800.) dso_internal { maxRho_ = r;}
  void setMaxDetectorLength(float l = 2200.) dso_internal { maxZ_ = l/2.;}
  void setMaxHORadius(float r = 800.) dso_internal { HOmaxRho_ = r;}
  void setMaxHOLength(float l = 2200.) dso_internal { HOmaxZ_ = l/2.;}
  void setMinDetectorRadius(float r = 0.)  dso_internal { minRho_ = r;}
  void setMinDetectorLength(float l = 0.)  dso_internal { minZ_ = l/2.;}
  
  void setPropagationStep(float s = 20.){ step_ = s;}
  float getPropagationStep() const  dso_internal { return step_;}
  
protected:
  
  static int sign (float number) dso_internal {
    if (number ==0) return 0;
    if (number > 0)
      return 1;
    else
      return -1;
  }
  
  std::pair<float,float> delta( const double& theta1,
				const double& theta2,
				const double& phi1,
				const double& phi2) dso_internal;
  
  float distance(const Plane* plane, int index) dso_internal {
    if (index<0 || fullTrajectory_.empty() || (unsigned int)index >= fullTrajectory_.size()) return 0;
    return plane->localZ(fullTrajectory_[index].position());
  }
  
  std::deque<SteppingHelixStateInfo> fullTrajectory_;
  std::vector<SteppingHelixStateInfo> ecalTrajectory_;
  std::vector<SteppingHelixStateInfo> hcalTrajectory_;
  std::vector<SteppingHelixStateInfo> hoTrajectory_;
  std::vector<SteppingHelixStateInfo> preshowerTrajectory_;
  std::vector<GlobalPoint> wideEcalTrajectory_; 
  std::vector<GlobalPoint> wideHcalTrajectory_;
  std::vector<GlobalPoint> wideHOTrajectory_;
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
