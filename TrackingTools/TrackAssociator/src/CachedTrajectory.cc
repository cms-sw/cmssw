// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      CachedTrajectory
// 
// $Id: CachedTrajectory.cc,v 1.5 2007/03/08 04:19:26 dmytro Exp $
//
//


#include "TrackingTools/TrackAssociator/interface/CachedTrajectory.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include <deque>
#include <algorithm>

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
	try {
	   state = shp->propagate(state, *target);
	}
	catch(...){
	   edm::LogWarning("TrackAssociator") << "An exception is caught during the track propagation\n"
	     << state.momentum().x() << ", " << state.momentum().y() << ", " << state.momentum().z();
	   state = SteppingHelixStateInfo();
	}
     }
   else
     {
	FreeTrajectoryState fts;
	state.getFreeState( fts );
	TrajectoryStateOnSurface stateOnSurface = propagator_->propagate(fts, *target);
	state = SteppingHelixStateInfo( *(stateOnSurface.freeState()) );
     }
   
   // LogTrace("TrackAssociator")
   // << state.position().mag() << " , "   << state.position().eta() << " , "
   // << state.position().phi();
}


void CachedTrajectory::propagateAll(const SteppingHelixStateInfo& initialState)
{
   if ( fullTrajectoryFilled_ ) {
      edm::LogWarning("TrackAssociator") << "Reseting all trajectories. Please call reset_trajectory() explicitely to avoid this message";
      reset_trajectory();
   }
	
   TimerStack timers(TimerStack::Disableable);

   reset_trajectory();
   if (propagator_==0) throw cms::Exception("FatalError") << "Track propagator is not defined\n";
   SteppingHelixStateInfo currentState(initialState);
   
   while (currentState.position().perp()<maxRho_ && fabs(currentState.position().z())<maxZ_ ){
      propagateForward(currentState,step_);
      if (! currentState.isValid() ) {
	 LogTrace("TrackAssociator") << "Failed to propagate the track; moving on\n";
	 break;
      }
      fullTrajectory_.push_back(currentState);
   }
   LogTrace("TrackAssociator") << "Done with the track propagation in the detector. Number of steps: " << fullTrajectory_.size();
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
      LogTrace("TrackAssociator") << "Track didn't cross the plane:\n\tleft distance: "<<distance(plane, leftIndex)
	<<"\n\tright distance: " << distance(plane, rightIndex);
     return TrajectoryStateOnSurface();
   }
   
   while (leftIndex + 1 < rightIndex) {
      closestPointOnLeft = int((leftIndex+rightIndex)/2);
      float dist = distance(plane,closestPointOnLeft);
      /*
      LogTrace("TrackAssociator") << "Closest point on left: " << closestPointOnLeft << "\n"
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
      LogTrace("TrackAssociator") << "Distance on left: " << distance(plane, leftIndex) << "\n"
	<< "Distance to closest point: " <<  distance(plane, closestPointOnLeft) << "\n"
	<< "Left index: " << leftIndex << "\n"
	<< "Right index: " << rightIndex;
       */
   }
   //   LogTrace("TrackAssociator") << "closestPointOnLeft: " << closestPointOnLeft 
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
	SteppingHelixStateInfo state;
	try { 
	   state = shp->propagate(fullTrajectory_[closestPointOnLeft], *plane);
	}
	catch(...){
	   edm::LogWarning("TrackAssociator") << "An exception is caught during the track propagation\n"
	     << state.momentum().x() << ", " << state.momentum().y() << ", " << state.momentum().z();
	   return TrajectoryStateOnSurface();
	}
	return state.getStateOnSurface(*plane);
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
				     const FiducialVolume& volume,
				     int steps)
{
   if ( ! fullTrajectoryFilled_ ) throw cms::Exception("FatalError") << "trajectory is not defined yet. Please use propagateAll first.";
   if ( fullTrajectory_.empty() ) edm::LogWarning("TrackAssociator") << "full cached trajectory is empty. This doesn't make sense";
	
   if ( ! volume.isValid() ) {
      LogTrace("TrackAssociator") << "no trajectory is expected to be found since the fiducial volume is not valid";
      return;
   }
   double step = std::max(volume.maxR()-volume.minR(),volume.maxZ()-volume.minZ())/steps;
   
   int closestPointOnLeft = -1;
   
   // check whether the trajectory crossed the region
   if ( ! 
	( ( fullTrajectory_.front().position().perp() < volume.maxR() && fabs(fullTrajectory_.front().position().z()) < volume.maxZ() ) &&
	  ( fullTrajectory_.back().position().perp() > volume.minR() || fabs(fullTrajectory_.back().position().z()) > volume.minZ() ) ))
     {
	LogTrace("TrackAssociator") << "Track didn't cross the region (R1,R2,L1,L2): " << volume.minR() << ", " << volume.maxR() <<
	  ", " << volume.minZ() << ", " << volume.maxZ();
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
      // LogTrace("TrackAssociator") << "Trajectory info (i,perp,r1,r2,z,z1,z2): " << i << ", " << fullTrajectory_[i].position().perp() <<
      //	", " << volume.minR() << ", " << volume.maxR() << ", " << fullTrajectory_[i].position().z() << ", " << volume.minZ() << ", " << 
      //	volume.maxZ() << ", " << closestPointOnLeft;
      if ( fullTrajectory_[i].position().perp()-volume.minR() > 0  || fabs(fullTrajectory_[i].position().z()) - volume.minZ() >0 )
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
   while (currentState.position().perp() < volume.maxR() && fabs(currentState.position().z()) < volume.maxZ() )
     {
	propagateForward(currentState,step);
	if (! currentState.isValid() ) {
	   LogTrace("TrackAssociator") << "Failed to propagate the track; moving on\n";
	   break;
	}
	// LogTrace("TrackAssociator") << "New state (perp, z): " << currentState.position().perp() << ", " << currentState.position().z();
	if ( ( currentState.position().perp() < volume.maxR() && fabs(currentState.position().z()) < volume.maxZ() ) &&
	     ( currentState.position().perp()-volume.minR() > 0  || fabs(currentState.position().z()) - volume.minZ() >0 ) )
	  trajectory.push_back(currentState);
     }
}

void CachedTrajectory::reset_trajectory() { 
   fullTrajectory_.clear();
   ecalTrajectory_.clear();
   hcalTrajectory_.clear();
   hoTrajectory_.clear();
   fullTrajectoryFilled_ = false;
}

void CachedTrajectory::findEcalTrajectory( const FiducialVolume& volume ) {
   LogTrace("TrackAssociator") << "getting trajectory in ECAL";
   getTrajectory(ecalTrajectory_, volume, 4 );
   LogTrace("TrackAssociator") << "# of points in ECAL trajectory:" << ecalTrajectory_.size();
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getEcalTrajectory() {
   return ecalTrajectory_;
}

void CachedTrajectory::findHcalTrajectory( const FiducialVolume& volume ) {
   LogTrace("TrackAssociator") << "getting trajectory in HCAL";
   getTrajectory(hcalTrajectory_, volume, 4 );
   LogTrace("TrackAssociator") << "# of points in HCAL trajectory:" << hcalTrajectory_.size();
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getHcalTrajectory() {
   return hcalTrajectory_;
}

void CachedTrajectory::findHOTrajectory( const FiducialVolume& volume ) {
   LogTrace("TrackAssociator") << "getting trajectory in HO";
   getTrajectory(hoTrajectory_, volume, 2 );
   LogTrace("TrackAssociator") << "# of points in HO trajectory:" << hoTrajectory_.size();
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getHOTrajectory() {
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
