#ifndef TrackAssociator_MuonDetIdAssociator_h
#define TrackAssociator_MuonDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      MuonDetIdAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: MuonDetIdAssociator.h,v 1.2 2006/08/25 17:35:40 jribnik Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/MuonChamberMatch.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class MuonDetIdAssociator: public DetIdAssociator{
 public:
   MuonDetIdAssociator():DetIdAssociator(48, 48 , 0.125),geometry_(0){};
   MuonDetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
     :DetIdAssociator(nPhi, nEta, etaBinSize),geometry_(0){};
   
   virtual void setGeometry(const GlobalTrackingGeometry* ptr){ geometry_ = ptr; }
   
   std::vector<MuonChamberMatch> getTrajectoryInMuonDetector(const FreeTrajectoryState& initialState,
							     const float dRMuonPreselection,
							     const float maxDistanceX,
							     const float maxDistanceY);
   virtual const GeomDet* getGeomDet( const DetId& id );

 protected:
   
   virtual void reset_trajectory() { trajectory_.clear(); }
   
   // fly through the whole muon detector
   virtual void propagateAll(const FreeTrajectoryState& initialState);
   
   // get fast to a given DetId surface using cached trajectory
   virtual TrajectoryStateOnSurface propagate(const DetId);
   
   // calculate trajectory size 
   virtual float trajectoryDeltaEta();
   virtual float trajectoryDeltaPhi();
   
   virtual void check_setup();
   
   virtual GlobalPoint getPosition(const DetId& id);
   
   virtual std::set<DetId> getASetOfValidDetIds();
   
   virtual std::vector<GlobalPoint> getDetIdPoints(const DetId& id);

   virtual bool insideElement(const GlobalPoint& point, const DetId& id);

   const GlobalTrackingGeometry* geometry_;
   
   static int sign (float number){
      if (number ==0) return 0;
      if (number == fabs(number))
	return 1;
      else
	return -1;
   }
   
   float distance(const Plane* plane, int index) {
      if (index<0 || trajectory_.empty() || uint(index) >= trajectory_.size()) return 0;
      return plane->localZ(trajectory_[index].freeState()->position());
   }
   
   std::vector<TrajectoryStateOnSurface> trajectory_;
};
#endif
