// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      CachedTrajectory
// 
// $Id: CachedTrajectory.cc,v 1.2 2006/12/19 06:39:33 dmytro Exp $
//
//


#include "TrackingTools/TrackAssociator/interface/CachedTrajectory.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "Geometry/Surface/interface/Plane.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include <deque>

CachedTrajectory::CachedTrajectory():propagator_(0){
   reset_trajectory();
   setDetectorRadius();
   setDetectorLength();
   setPropagationStep();
}

void CachedTrajectory::propagateForward(SteppingHelixStateInfo& state, float distance)
{
   // defined a normal plane wrt the particle trajectory direction
   // let's hope that I computed the rotation matrix correctly.
   GlobalVector vector(state.momentum().unit());
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
   Surface* target = new Plane(state.position()+vector*distance, rotation);
   if( SteppingHelixPropagator* shp = dynamic_cast<SteppingHelixPropagator*>(propagator_) )
     {
	state = shp->propagate(state, *target);
     }
   else
     {
	FreeTrajectoryState fts;
	state.getFreeState( fts );
	TrajectoryStateOnSurface stateOnSurface = propagator_->propagate(fts, *target);
	state = SteppingHelixStateInfo( *(stateOnSurface.freeState()) );
     }
   

   // LogTrace("CachedTrajectory") << "\ttrajectory point (z,mag,eta,phi): " << state.position().z() << ", "
   // << state.position().mag() << " , "   << state.position().eta() << " , "
   // << state.position().phi();
}


void CachedTrajectory::propagateAll(const SteppingHelixStateInfo& initialState)
{
   if ( fullTrajectoryFilled_ ) {
      edm::LogWarning("") << "Reseting all trajectories. Please call reset_trajectory() explicitely to avoid this message";
      reset_trajectory();
   }
	
   TimerStack timers(TimerStack::Disableable);

   reset_trajectory();
   if (propagator_==0) throw cms::Exception("FatalError") << "Track propagator is not defined\n";
   SteppingHelixStateInfo currentState(initialState);
   
   while (currentState.position().perp()<maxRho_ && fabs(currentState.position().z())<maxZ_ ){
      propagateForward(currentState,step_);
      if (! currentState.isValid() ) {
	 LogTrace("FailedPropagation") << "Failed to propagate the track; moving on\n";
	 break;
      }
      fullTrajectory_.push_back(currentState);
   }
   LogTrace("") << "Done with the track propagation in the detector. Number of steps: " << fullTrajectory_.size();
   fullTrajectoryFilled_ = true;
}

TrajectoryStateOnSurface CachedTrajectory::propagate(const Plane* plane)
{
   TimerStack timers(TimerStack::Disableable);
   // timers.benchmark("CachedTrajectory::propagate::benchmark");
   timers.push("CachedTrajectory::propagate",TimerStack::FastMonitoring);
   timers.push("CachedTrajectory::propagate::findClosestPoint",TimerStack::FastMonitoring);

   // Assume that all points along the trajectory are equally spread out.
   // For simplication assume that the trajectory is just a straight
   // line and find a point closest to the target plane. Propagate to
   // the plane from the point.
   
   const float matchingDistance = 1;
   // find the closest point to the plane
   int leftIndex = 0;
   int rightIndex = fullTrajectory_.size()-1;
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
   //     << fullTrajectory_[closestPointOnLeft].freeState()->position().z() << ", "
   //     << fullTrajectory_[closestPointOnLeft].freeState()->position().perp() << " , "	
   //     << fullTrajectory_[closestPointOnLeft].freeState()->position().eta() << " , " 
   //     << fullTrajectory_[closestPointOnLeft].freeState()->position().phi()
   //     << "\n\tplane center (z,R,eta,phi): " 
   //     << plane->position().z() << ", "
   //     << plane->position().perp() << " , "	
   //     << plane->position().eta() << " , " 
   //     << plane->position().phi();
     
   // propagate to the plane
   timers.pop_and_push("CachedTrajectory::propagate::localPropagation",TimerStack::FastMonitoring);
   if (SteppingHelixPropagator* shp = dynamic_cast<SteppingHelixPropagator*>(propagator_))
     {
	SteppingHelixStateInfo shsi(shp->propagate(fullTrajectory_[closestPointOnLeft], *plane));
	return shsi.getStateOnSurface(*plane);
     }
   else
     {
	FreeTrajectoryState fts;
	fullTrajectory_[closestPointOnLeft].getFreeState(fts);
	return propagator_->propagate(fts, *plane);
     }
}

float CachedTrajectory::trajectoryDeltaEta()
{
   
   // we are not interested in the inner part of the detector if for some reason 
   // the track was not propagated through calorimeters.
   // if(newState.position().mag() > 300) 

   float minEta = 99999;
   float maxEta = -99999;
   for(std::vector<SteppingHelixStateInfo>::const_iterator point = fullTrajectory_.begin();
       point != fullTrajectory_.end(); point++){
      if (point->position().eta() > maxEta) maxEta = point->position().eta();
      if (point->position().eta() < minEta) minEta = point->position().eta();
   }
   if (minEta>maxEta) return 0;
   return maxEta-minEta;
}

float CachedTrajectory::trajectoryDeltaPhi()
{
   float minPhi = 99999;
   float maxPhi = -99999;
   for(std::vector<SteppingHelixStateInfo>::const_iterator point = fullTrajectory_.begin();
       point != fullTrajectory_.end(); point++){
      if (point->position().phi() > maxPhi) maxPhi = point->position().phi();
      if (point->position().phi() < minPhi) minPhi = point->position().phi();
   }
   if (minPhi>maxPhi) return 0;
   // assuming that we are not reconstructing loopers, so dPhi should be small compared with 2Pi
   if (minPhi+2*3.1415926-maxPhi<maxPhi-minPhi)
     return minPhi+2*3.1415926-maxPhi;
   else
     return maxPhi-minPhi;
}

void CachedTrajectory::getTrajectory(std::vector<SteppingHelixStateInfo>& trajectory,
				     const float r1, 
				     const float r2, 
				     const float z1, 
				     const float z2, 
				     const float step)
{
   if ( ! fullTrajectoryFilled_ ) throw cms::Exception("FatalError") << "trajectory is not defined yet. Please use propagateAll first.";
	
   if (r1>r2 || z1>z2) {
      LogTrace("CachedTrajectory") << "no trajectory is expected to be found since either R1>R2 or L1>L2";
      return;
   }
   
   int closestPointOnLeft = -1;
   
   // check whether the trajectory crossed the region
   if ( ! 
	( ( fullTrajectory_.front().position().perp()<r2 && fabs(fullTrajectory_.front().position().z()) <z2 ) &&
	  ( fullTrajectory_.back().position().perp()>r1  || fabs(fullTrajectory_.back().position().z())  >z1 ) ))
     {
	LogTrace("") << "Track didn't cross the region (R1,R2,L1,L2): " << r1 << ", " << r2 << ", " << z1 << ", " <<z2;
	return;
     }
   
   // get distance along momentum to the surface.
   
   // the following code can be made faster, but it'll hardly be a significant improvement
   // simplifications:
   //   1) direct loop over stored trajectory points instead of some sort 
   //      of fast root search (Newton method)
   //   2) propagate from the closest point outside the region with the 
   //      requested step ignoring stored trajectory points.
   for(uint i=0; i<fullTrajectory_.size(); i++) {
      LogTrace("") << "Trajectory info (i,perp,r1,r2,z,z1,z2): " << i << ", " << fullTrajectory_[i].position().perp() <<
	", " << r1 << ", " << r2 << ", " << fullTrajectory_[i].position().z() << ", " << z1 << ", " << z2 <<
	", " << closestPointOnLeft;
      if ( fullTrajectory_[i].position().perp()-r1>0  || fabs(fullTrajectory_[i].position().z()) - z1 >0 )
	{
	   if (i>0) 
	     closestPointOnLeft = i - 1;
	   else
	     closestPointOnLeft = 0;
	   break;
	}
   }
   if (closestPointOnLeft == -1) throw cms::Exception("FatalError") << "This shouls never happen - internal logic error";
   
   SteppingHelixStateInfo currentState(fullTrajectory_[closestPointOnLeft]);
   while (currentState.position().perp()<r2 && fabs(currentState.position().z()) <z2 )
     {
	propagateForward(currentState,step);
	if (! currentState.isValid() ) {
	   LogTrace("FailedPropagation") << "Failed to propagate the track; moving on\n";
	   break;
	}
	if ( ( currentState.position().perp()<r2 && fabs(currentState.position().z()) < z2 ) &&
	     ( currentState.position().perp()-r1>0  || fabs(currentState.position().z()) - z1 >0 ) )
	  trajectory.push_back(currentState);
     }
}

void CachedTrajectory::reset_trajectory() { 
   fullTrajectory_.clear();
   ecalTrajectory_.clear();
   hcalTrajectory_.clear();
   hoTrajectory_.clear();
   fullTrajectoryFilled_ = false;
   ecalTrajectoryFilled_ = false;
   hcalTrajectoryFilled_ = false;
   hoTrajectoryFilled_ = false;
}


const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getEcalTrajectory() {
   if ( ! ecalTrajectoryFilled_ )
     {
	LogTrace("") << "getting trajectory in ECAL";
	getTrajectory(ecalTrajectory_, 130.,150.,315.,335,10 );
	LogTrace("") << "# of points in ECAL trajectory:" << ecalTrajectory_.size();
	ecalTrajectoryFilled_ = true;
     }
   return ecalTrajectory_;
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getHcalTrajectory() {
   if ( ! hcalTrajectoryFilled_ ) 
     {
	LogTrace("") << "getting trajectory in HCAL";
	getTrajectory(hcalTrajectory_, 190., 240., 400., 550, 50 );
	LogTrace("") << "# of points in HCAL trajectory:" << hcalTrajectory_.size();
	hcalTrajectoryFilled_ = true;
     }
   return hcalTrajectory_;
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getHOTrajectory() {
   if ( ! hoTrajectoryFilled_ ) 
     { 
	LogTrace("") << "getting trajectory in HO";
	getTrajectory(hoTrajectory_, 380,420.,625.,625.,10 );
	LogTrace("") << "# of points in HO trajectory:" << hoTrajectory_.size();
	hoTrajectoryFilled_ = true;
     }
   return hoTrajectory_;
}
   
SteppingHelixStateInfo CachedTrajectory::getStateAtEcal()
{
   if ( ecalTrajectory_.empty() )
     return SteppingHelixStateInfo();
   else 
     return ecalTrajectory_.front();
}

SteppingHelixStateInfo CachedTrajectory::getStateAtHcal()
{
   if ( hcalTrajectory_.empty() )
     return SteppingHelixStateInfo();
   else 
     return hcalTrajectory_.front();
}

SteppingHelixStateInfo CachedTrajectory::getStateAtHO()
{
   if ( hoTrajectory_.empty() )
     return SteppingHelixStateInfo();
   else 
     return hoTrajectory_.front();
}
