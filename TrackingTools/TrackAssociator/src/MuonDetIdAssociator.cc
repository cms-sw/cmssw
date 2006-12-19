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
// $Id: DetIdAssociator.cc,v 1.6 2006/10/20 16:36:22 dmytro Exp $
//
//


#include "TrackingTools/TrackAssociator/interface/MuonDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"
#include "Geometry/Surface/interface/Plane.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include <deque>

void MuonDetIdAssociator::check_setup(){
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   DetIdAssociator::check_setup();
}

const GeomDet* MuonDetIdAssociator::getGeomDet( const DetId& id )
{
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   const GeomDet* gd = geometry_->idToDet(id);
   if (gd == 0) throw cms::Exception("NoGeometry") << "Cannot find GeomDet for DetID: " << id.rawId() <<"\n";
   return gd;
}


GlobalPoint MuonDetIdAssociator::getPosition(const DetId& id){
   Surface::PositionType point(getGeomDet(id)->surface().position());
   return GlobalPoint(point.x(),point.y(),point.z());
}

std::set<DetId> MuonDetIdAssociator::getASetOfValidDetIds(){
   std::set<DetId> setOfValidIds;
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   // we need to store only DTChambers as well as CSCChambers
   // Let's get all GeomDet by dets() and select only DTChambers and CSCChambers
   std::vector<GeomDet*> vectOfGeomDetPtrs = geometry_->dets();
   LogTrace("MuonDetIdAssociator::getASetOfValidDetIds") << "Number of GeomDet found: " << vectOfGeomDetPtrs.size();
   for(std::vector<GeomDet*>::const_iterator it = vectOfGeomDetPtrs.begin(); it != vectOfGeomDetPtrs.end(); ++it)
     {
	if ((*it)->subDetector() == GeomDetEnumerators::CSC || (*it)->subDetector() == GeomDetEnumerators::DT)
	  {
	     if (DTChamber* dt = dynamic_cast< DTChamber*>(*it)) {
		setOfValidIds.insert(dt->id());
	     }else{
		if (CSCChamber* csc = dynamic_cast< CSCChamber*>(*it)) {
		   setOfValidIds.insert(csc->id());
		}
	     }
	  }
     }
   return setOfValidIds;
}

bool MuonDetIdAssociator::insideElement(const GlobalPoint& point, const DetId& id){
   if (geometry_==0) throw cms::Exception("ConfigurationProblem") << "GlobalTrackingGeomtry is not set\n";
   LocalPoint lp = geometry_->idToDet(id)->toLocal(point);
   return geometry_->idToDet(id)->surface().bounds().inside(lp);
}

std::vector<GlobalPoint> MuonDetIdAssociator::getDetIdPoints(const DetId& id){
   std::vector<GlobalPoint> points;
   const GeomDet* geomDet = getGeomDet( id );
   
   // the coners of muon detector elements are not stored and can be only calculated
   // based on methods defined in the interface class Bounds:
   //   width() - x
   //   length() - y 
   //   thinkness() - z
   // NOTE: this convention is implementation specific and can fail. Both
   //       RectangularPlaneBounds and TrapezoidalPlaneBounds use it.
   // Even though the CSC geomtry is more complicated (trapezoid),  it's enough 
   // to estimate which bins should contain this element. For the distance
   // calculation from the edge, we will use exact geometry to get it right.
	    
   const Bounds* bounds = &(geometry_->idToDet(id)->surface().bounds());
   points.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,+bounds->length()/2,+bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,+bounds->length()/2,+bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,-bounds->length()/2,+bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,-bounds->length()/2,+bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,+bounds->length()/2,-bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,+bounds->length()/2,-bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(+bounds->width()/2,-bounds->length()/2,-bounds->thickness()/2)));
   points.push_back(geomDet->toGlobal(LocalPoint(-bounds->width()/2,-bounds->length()/2,-bounds->thickness()/2)));
   
   return  points;
}

std::vector<MuonChamberMatch> MuonDetIdAssociator::getTrajectoryInMuonDetector(const FreeTrajectoryState& initialState,
									       const float dRMuonPreselection,
									       const float maxDistanceX,
									       const float maxDistanceY)
{
   // Strategy:
   //    Propagate through the whole detector, estimate change in eta and phi 
   //    along the trajectory, add this to dRMuon and find DetIds around this 
   //    direction using the map. Then propagate fast to each surface and apply 
   //    final matching criteria.

   TimerStack timers(TimerStack::Disableable);
   timers.push("MuonDetIdAssociator::getTrajectoryInMuonDetector");
   check_setup();
   std::vector<MuonChamberMatch> matches;
   timers.push("MuonDetIdAssociator::getTrajectoryInMuonDetector::propagation",TimerStack::FastMonitoring);
   propagateAll(initialState);
   timers.pop();
   if(! trajectory_.empty() ) {
      // get DetIds
      float dEta = trajectoryDeltaEta();
      float dPhi = trajectoryDeltaPhi();
      float lookUpCone = ( dEta > dPhi ? dEta : dPhi ) + dRMuonPreselection;
      timers.push("MuonDetIdAssociator::getTrajectoryInMuonDetector::getDetIdsCloseToAPoint",TimerStack::FastMonitoring);
      std::set<DetId> muonIdsInRegion = getDetIdsCloseToAPoint(trajectory_[0].freeState()->position(), lookUpCone);
      timers.pop_and_push("MuonDetIdAssociator::getTrajectoryInMuonDetector::matching",TimerStack::FastMonitoring);
      LogTrace("getTrajectoryInMuonDetector") << "Number of chambers to check: " << muonIdsInRegion.size();
	
      for(std::set<DetId>::const_iterator detId = muonIdsInRegion.begin(); detId != muonIdsInRegion.end(); detId++)
	{
	   timers.push("MuonDetIdAssociator::getTrajectoryInMuonDetector::matching::localPropagation",TimerStack::FastMonitoring);
	   TrajectoryStateOnSurface stateOnSurface = propagate(*detId);
	   if (! stateOnSurface.isValid()) {
	      LogTrace("FailedPropagation") << "Failed to propagate the track; moving on\n\t"<<
		detId->rawId() << " not crossed\n"; ;
	      continue;
	   }
	   timers.pop_and_push("MuonDetIdAssociator::getTrajectoryInMuonDetector::matching::geometryAccess",TimerStack::FastMonitoring);
	   const GeomDet* geomDet = getGeomDet(*detId);
	   LocalPoint localPoint = geomDet->surface().toLocal(stateOnSurface.freeState()->position());
	   float distanceX = fabs(localPoint.x()) - geomDet->surface().bounds().width()/2;
	   float distanceY = fabs(localPoint.y()) - geomDet->surface().bounds().length()/2;
	   timers.pop_and_push("MuonDetIdAssociator::getTrajectoryInMuonDetector::matching::checking",TimerStack::FastMonitoring);
	   if (distanceX < maxDistanceX && distanceY < maxDistanceY) {
	      LogTrace("getTrajectoryInMuonDetector") << "found a match, DetId: " << detId->rawId();
	      MuonChamberMatch match;
	      match.tState = stateOnSurface;
	      match.localDistanceX = distanceX;
	      match.localDistanceY = distanceY;
	      match.id = *detId;
	      matches.push_back(match);
	   }
	   timers.pop();
	}
      timers.pop();

   }
   return matches;
}

void MuonDetIdAssociator::propagateAll(const FreeTrajectoryState& initialState)
{
   TimerStack timers(TimerStack::Disableable);
   const float maxRho(800.); 
   const float maxZ(1100.);
   const float step(20.);

   check_setup();
   reset_trajectory();
   FreeTrajectoryState current(initialState);
   const Surface* target(0);
   
   while (current.position().perp()<maxRho && fabs(current.position().z())<maxZ ){
      // defined a normal plane wrt the particle trajectory direction
      // let's hope that I computed the rotation matrix correctly.
      GlobalVector vector(current.momentum().unit());
      float r21 = 0;
      float r22 = vector.z()/sqrt(1-pow(vector.x(),2));
      float r23 = -vector.y()/sqrt(1-pow(vector.x(),2));
      float r31 = vector.x();
      float r32 = vector.y();
      float r33 = vector.z();
      float r11 = r22*r33-r23*r32;
      float r12 = r23*r31;
      float r13 = -r22*r31;
      Surface::RotationType rotation(r11, r12, r13,
				     r21, r22, r23,
				     r31, r32, r33);
      target = new Plane(current.position()+vector*step, rotation);
      timers.push("MuonDetIdAssociator::propagateAll",TimerStack::FastMonitoring);
      TrajectoryStateOnSurface stateOnSurface = ivProp_->propagate(current, *target);
      timers.pop();
      if (! stateOnSurface.isValid()) {
         LogTrace("FailedPropagation") << "Failed to propagate the track; moving on\n";
         break;
      }
      // LogTrace("MuonDetIdAssociator") << "\ttrajectory point (z,mag,eta,phi): " << stateOnSurface.freeState()->position().z() << ", "
      // << stateOnSurface.freeState()->position().mag() << " , "   << stateOnSurface.freeState()->position().eta() << " , "
      // << stateOnSurface.freeState()->position().phi();
      //
      // we are not interested in the inner part of the detector if for some reason 
      // the track was not propagated through calorimeters.
      if(stateOnSurface.freeState()->position().mag() > 300) trajectory_.push_back(stateOnSurface);
      current = FreeTrajectoryState(*(stateOnSurface.freeState()));
   }
   LogTrace("") << "Done with track propagation in the muon detector. Number of steps: " << trajectory_.size();
}

TrajectoryStateOnSurface MuonDetIdAssociator::propagate(const DetId detId)
{
   TimerStack timers(TimerStack::Disableable);
   // timers.benchmark("MuonDetIdAssociator::propagate::benchmark");
   timers.push("MuonDetIdAssociator::propagate",TimerStack::FastMonitoring);
   timers.push("MuonDetIdAssociator::propagate::findClosestPoint",TimerStack::FastMonitoring);

   // Assume that all points along the trajectory are equally spread out.
   // For simplication assume that the trajectory is just a straight
   // line and find a point closest to the given DetId plane. Propagate to
   // the plane from the point.
   
   // get the plane
   const Plane* plane = &(getGeomDet(detId)->surface());
   const float matchingDistance = 1;
   // find the closest point to the plane
   int leftIndex = 0;
   int rightIndex = trajectory_.size()-1;
   int closestPointOnLeft = 0;
   
   // check whether the trajectory crossed the plane (signs should be different)
   if ( sign( distance(plane, leftIndex) ) * sign( distance(plane, rightIndex) ) != -1 ) {
      LogTrace("") << "Track didn't cross the plane:\n\tleft distance: "<<distance(plane, leftIndex)
	<<"\n\tright distance: " << distance(plane, rightIndex);
     return TrajectoryStateOnSurface();
   }
   
   while (leftIndex + 1 < rightIndex) {
      closestPointOnLeft = int((leftIndex+rightIndex)/2);
      float dist = distance(plane,closestPointOnLeft);
      /*
      LogTrace("") << "Closest point on left: " << closestPointOnLeft << "\n"
	<< "Distance to the plane: " << dist; */
      if (fabs(dist)<matchingDistance) {
	 // found close match, verify that we are on the left side
	 if (closestPointOnLeft>0 && sign( distance(plane, closestPointOnLeft-1) ) * dist == -1)
	   closestPointOnLeft--;
	 break; 
      }
      
      // check where is the plane
      if (sign( distance(plane, leftIndex) * dist ) == -1)
	rightIndex = closestPointOnLeft;
      else
	leftIndex = closestPointOnLeft;
      /*
      LogTrace("") << "Distance on left: " << distance(plane, leftIndex) << "\n"
	<< "Distance to closest point: " <<  distance(plane, closestPointOnLeft) << "\n"
	<< "Left index: " << leftIndex << "\n"
	<< "Right index: " << rightIndex;
       */
   }
   //   LogTrace("") << "closestPointOnLeft: " << closestPointOnLeft 
   //     << "\n\ttrajectory point (z,R,eta,phi): " 
   //     << trajectory_[closestPointOnLeft].freeState()->position().z() << ", "
   //     << trajectory_[closestPointOnLeft].freeState()->position().perp() << " , "	
   //     << trajectory_[closestPointOnLeft].freeState()->position().eta() << " , " 
   //     << trajectory_[closestPointOnLeft].freeState()->position().phi()
   //     << "\n\tplane center (z,R,eta,phi): " 
   //     << plane->position().z() << ", "
   //     << plane->position().perp() << " , "	
   //     << plane->position().eta() << " , " 
   //     << plane->position().phi();
     
   // propagate to the plane
   timers.pop_and_push("MuonDetIdAssociator::propagate::localPropagation",TimerStack::FastMonitoring);
   return ivProp_->propagate(*(trajectory_[closestPointOnLeft].freeState()), *plane);
}

float MuonDetIdAssociator::trajectoryDeltaEta()
{
   float minEta = 99999;
   float maxEta = -99999;
   for(std::vector<TrajectoryStateOnSurface>::const_iterator point = trajectory_.begin();
       point != trajectory_.end(); point++){
      if (point->freeState()->position().eta() > maxEta) maxEta = point->freeState()->position().eta();
      if (point->freeState()->position().eta() < minEta) minEta = point->freeState()->position().eta();
   }
   if (minEta>maxEta) return 0;
   return maxEta-minEta;
}

float MuonDetIdAssociator::trajectoryDeltaPhi()
{
   float minPhi = 99999;
   float maxPhi = -99999;
   for(std::vector<TrajectoryStateOnSurface>::const_iterator point = trajectory_.begin();
       point != trajectory_.end(); point++){
      if (point->freeState()->position().phi() > maxPhi) maxPhi = point->freeState()->position().phi();
      if (point->freeState()->position().phi() < minPhi) minPhi = point->freeState()->position().phi();
   }
   if (minPhi>maxPhi) return 0;
   // assuming that we are not reconstructing loopers, so dPhi should be small compared with 2Pi
   if (minPhi+2*3.1415926-maxPhi<maxPhi-minPhi)
     return minPhi+2*3.1415926-maxPhi;
   else
     return maxPhi-minPhi;
}



