// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      CachedTrajectory
// 
// $Id: CachedTrajectory.cc,v 1.31 2012/12/25 16:07:26 innocent Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/CachedTrajectory.h"
// #include "Utilities/Timing/interface/TimerStack.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include <deque>
#include <algorithm>

std::vector<SteppingHelixStateInfo> 
propagateThoughFromIP(const SteppingHelixStateInfo& state,const Propagator* prop,
		      const FiducialVolume& volume,int nsteps,
		      float step, float minR, float minZ, float maxR, float maxZ) {
   CachedTrajectory neckLace;
   neckLace.setStateAtIP(state);
   neckLace.reset_trajectory();
   neckLace.setPropagator(prop);
   neckLace.setPropagationStep(0.1);
   neckLace.setMinDetectorRadius(minR);
   neckLace.setMinDetectorLength(minZ*2.);
   neckLace.setMaxDetectorRadius(maxR);
   neckLace.setMaxDetectorLength(maxZ*2.);

   // Propagate track
   bool isPropagationSuccessful = neckLace.propagateAll(state);

   if (!isPropagationSuccessful)
     return std::vector<SteppingHelixStateInfo> () ;

   std::vector<SteppingHelixStateInfo> complicatePoints;
   neckLace.getTrajectory(complicatePoints, volume, nsteps);

   return complicatePoints;

}



CachedTrajectory::CachedTrajectory():propagator_(0){
   reset_trajectory();
   setMaxDetectorRadius();
   setMaxDetectorLength();
   setMinDetectorRadius();
   setMinDetectorLength();
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
   Plane::PlanePointer target = Plane::build(state.position()+vector*distance, rotation);
   propagate(state, *target);
}

void CachedTrajectory::propagate(SteppingHelixStateInfo& state, const Plane& plane)
{
   if( const SteppingHelixPropagator* shp = dynamic_cast<const SteppingHelixPropagator*>(propagator_) )
     {
	try {
	   state = shp->propagate(state, plane);
	}
	catch(cms::Exception &ex){
           edm::LogWarning("TrackAssociator") << 
                "Caught exception " << ex.category() << ": " << ex.explainSelf();
	   edm::LogWarning("TrackAssociator") << "An exception is caught during the track propagation\n"
	     << state.momentum().x() << ", " << state.momentum().y() << ", " << state.momentum().z();
	   state = SteppingHelixStateInfo();
	}
     }
   else
     {
	FreeTrajectoryState fts;
	state.getFreeState( fts );
	TrajectoryStateOnSurface stateOnSurface = propagator_->propagate(fts, plane);
	state = SteppingHelixStateInfo( *(stateOnSurface.freeState()) );
     }
}

void CachedTrajectory::propagate(SteppingHelixStateInfo& state, const Cylinder& cylinder)
{
   if( const SteppingHelixPropagator* shp = dynamic_cast<const SteppingHelixPropagator*>(propagator_) )
     {
	try {
	   state = shp->propagate(state, cylinder);
	}
	catch(cms::Exception &ex){
           edm::LogWarning("TrackAssociator") << 
                "Caught exception " << ex.category() << ": " << ex.explainSelf();
	   edm::LogWarning("TrackAssociator") << "An exception is caught during the track propagation\n"
	     << state.momentum().x() << ", " << state.momentum().y() << ", " << state.momentum().z();
	   state = SteppingHelixStateInfo();
	}
     }
   else
     {
	FreeTrajectoryState fts;
	state.getFreeState( fts );
	TrajectoryStateOnSurface stateOnSurface = propagator_->propagate(fts, cylinder);
	state = SteppingHelixStateInfo( *(stateOnSurface.freeState()) );
     }
}

bool CachedTrajectory::propagateAll(const SteppingHelixStateInfo& initialState)
{
   if ( fullTrajectoryFilled_ ) {
      edm::LogWarning("TrackAssociator") << "Reseting all trajectories. Please call reset_trajectory() explicitely to avoid this message";
      reset_trajectory();
   }
	
//   TimerStack timers(TimerStack::Disableable);

   reset_trajectory();
   if (propagator_==0) throw cms::Exception("FatalError") << "Track propagator is not defined\n";
   SteppingHelixStateInfo currentState(initialState);
   fullTrajectory_.push_back(currentState);

   while (currentState.position().perp()<maxRho_ && fabs(currentState.position().z())<maxZ_ ){
      LogTrace("TrackAssociator") << "[propagateAll] Propagate outward from (rho, r, z, phi) (" << 
	currentState.position().perp() << ", " << currentState.position().mag() << ", " <<
	currentState.position().z() << ", " << currentState.position().phi() << ")";
      propagateForward(currentState,step_);
     if (! currentState.isValid() ) {
       LogTrace("TrackAssociator") << "Failed to propagate the track; moving on\n";
       break;
     }
      LogTrace("TrackAssociator") << "\treached (rho, r, z, phi) (" << 
	currentState.position().perp() << ", " << currentState.position().mag() << ", " <<
	currentState.position().z() << ", " << currentState.position().phi() << ")";
     fullTrajectory_.push_back(currentState);
   }


   SteppingHelixStateInfo currentState2(initialState);
   SteppingHelixStateInfo previousState;
   while (currentState2.position().perp()>minRho_ || fabs(currentState2.position().z())>minZ_) {
      previousState=currentState2;
      propagateForward(currentState2,-step_);
      if (! currentState2.isValid() ) {
	 LogTrace("TrackAssociator") << "Failed to propagate the track; moving on\n";
	 break;
      }
      if(previousState.position().perp()- currentState2.position().perp() < 0) { 
	 LogTrace("TrackAssociator") << "Error: TrackAssociator has propogated the particle past the point of closest approach to IP" << std::endl;
	 break;
      }
      LogTrace("TrackAssociator") << "[propagateAll] Propagated inward from (rho, r, z, phi) (" << 
	previousState.position().perp() << ", " << previousState.position().mag() << ", " <<
	previousState.position().z() << "," << previousState.position().phi() << ") to (" << 
	currentState2.position().perp() << ", " << currentState2.position().mag() << ", " <<
	currentState2.position().z() << ", " << currentState2.position().phi() << ")";
      fullTrajectory_.push_front(currentState2);
   }




   // LogTrace("TrackAssociator") << "fullTrajectory_ has " << fullTrajectory_.size() << " states with (R, z):\n";
   // for(unsigned int i=0; i<fullTrajectory_.size(); i++) {
   //  LogTrace("TrackAssociator") << "state " << i << ": (" << fullTrajectory_[i].position().perp() << ", "
   //    << fullTrajectory_[i].position().z() << ")\n";
   // }





   LogTrace("TrackAssociator") << "Done with the track propagation in the detector. Number of steps: " << fullTrajectory_.size();
   fullTrajectoryFilled_ = true;
   return ! fullTrajectory_.empty();
}

TrajectoryStateOnSurface CachedTrajectory::propagate(const Plane* plane)
{
   // TimerStack timers(TimerStack::Disableable);
   // timers.benchmark("CachedTrajectory::propagate::benchmark");
   // timers.push("CachedTrajectory::propagate",TimerStack::FastMonitoring);
   // timers.push("CachedTrajectory::propagate::findClosestPoint",TimerStack::FastMonitoring);

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
      // LogTrace("TrackAssociator") << "Closest point on left: " << closestPointOnLeft << "\n"
      //    << "Distance to the plane: " << dist;
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
      
      // LogTrace("TrackAssociator") << "Distance on left: " << distance(plane, leftIndex) << "\n"
      //	<< "Distance to closest point: " <<  distance(plane, closestPointOnLeft) << "\n"
      //	<< "Left index: " << leftIndex << "\n"
      //	<< "Right index: " << rightIndex;
      
   }
      LogTrace("TrackAssociator") << "closestPointOnLeft: " << closestPointOnLeft 
        << "\n\ttrajectory point (z,R,eta,phi): " 
        << fullTrajectory_[closestPointOnLeft].position().z() << ", "
        << fullTrajectory_[closestPointOnLeft].position().perp() << " , "	
        << fullTrajectory_[closestPointOnLeft].position().eta() << " , " 
        << fullTrajectory_[closestPointOnLeft].position().phi()
        << "\n\tplane center (z,R,eta,phi): " 
        << plane->position().z() << ", "
        << plane->position().perp() << " , "	
        << plane->position().eta() << " , " 
        << plane->position().phi();
     
   // propagate to the plane
   // timers.pop_and_push("CachedTrajectory::propagate::localPropagation",TimerStack::FastMonitoring);
   if (const SteppingHelixPropagator* shp = dynamic_cast<const SteppingHelixPropagator*>(propagator_))
     {
	SteppingHelixStateInfo state;
	try { 
	   state = shp->propagate(fullTrajectory_[closestPointOnLeft], *plane);
	}
	catch(cms::Exception &ex){
           edm::LogWarning("TrackAssociator") << 
                "Caught exception " << ex.category() << ": " << ex.explainSelf();
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

std::pair<float,float> CachedTrajectory::trajectoryDelta( TrajectorType trajectoryType )
{
   // MEaning of trajectory change depends on its usage. In most cases we measure 
   // change in a trajectory as difference between final track position and initial 
   // direction. In some cases such as change of trajectory in the muon detector we 
   // might want to compare theta-phi of two points or even find local maximum and
   // mimimum. In general it's not essential what defenition of the trajectory change
   // is used since we use these numbers only as a rough estimate on how much wider
   // we should make the preselection region.
   std::pair<float,float> result(0,0);
   if ( ! stateAtIP_.isValid() ) { 
      edm::LogWarning("TrackAssociator") << "State at IP is not known set. Cannot estimate trajectory change. " <<
	"Trajectory change is not taken into account in matching";
      return result;
   }
   switch (trajectoryType) {
    case IpToEcal:
      if ( ecalTrajectory_.empty() )
	edm::LogWarning("TrackAssociator") << "ECAL trajector is empty. Cannot estimate trajectory change. " <<
	"Trajectory change is not taken into account in matching";
      else return delta( stateAtIP_.momentum().theta(), ecalTrajectory_.front().position().theta(), 
			 stateAtIP_.momentum().phi(), ecalTrajectory_.front().position().phi() );
      break;
    case IpToHcal:
      if ( hcalTrajectory_.empty() )
	edm::LogWarning("TrackAssociator") << "HCAL trajector is empty. Cannot estimate trajectory change. " <<
	"Trajectory change is not taken into account in matching";
      else return delta( stateAtIP_.momentum().theta(), hcalTrajectory_.front().position().theta(), 
			 stateAtIP_.momentum().phi(),   hcalTrajectory_.front().position().phi() );
      break;
    case IpToHO:
      if ( hoTrajectory_.empty() )
	edm::LogWarning("TrackAssociator") << "HO trajector is empty. Cannot estimate trajectory change. " <<
	"Trajectory change is not taken into account in matching";
      else return delta( stateAtIP_.momentum().theta(), hoTrajectory_.front().position().theta(), 
			 stateAtIP_.momentum().phi(),   hoTrajectory_.front().position().phi() );
      break;
    case FullTrajectory:
      if ( fullTrajectory_.empty() )
	edm::LogWarning("TrackAssociator") << "Full trajector is empty. Cannot estimate trajectory change. " <<
	"Trajectory change is not taken into account in matching";
      else  return delta( stateAtIP_.momentum().theta(), fullTrajectory_.back().position().theta(), 
			  stateAtIP_.momentum().phi(),   fullTrajectory_.back().position().phi() );
      break;
    default:
      edm::LogWarning("TrackAssociator") << "Unkown or not supported trajector type. Cannot estimate trajectory change. " <<
	"Trajectory change is not taken into account in matching";
   }
   return result;
}

std::pair<float,float> CachedTrajectory::delta(const double& theta1,
					       const double& theta2,
					       const double& phi1,
					       const double& phi2)
{
   std::pair<float,float> result(theta2 - theta1, phi2 - phi1 );
   // this won't work for loopers, since deltaPhi cannot be larger than Pi.
   if ( fabs(result.second) > 2*M_PI-fabs(result.second) ) {
      if (result.second>0) 
	result.second -= 2*M_PI;
      else
	result.second += 2*M_PI;
   }
   return result;
}
	
void CachedTrajectory::getTrajectory(std::vector<SteppingHelixStateInfo>& trajectory,
				     const FiducialVolume& volume,
				     int steps)
{
   if ( ! fullTrajectoryFilled_ ) throw cms::Exception("FatalError") << "trajectory is not defined yet. Please use propagateAll first.";
   if ( fullTrajectory_.empty() ) {
      LogTrace("TrackAssociator") << "Trajectory is empty. Move on";
      return;
   }
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
   double dZ(-1.);
   double dR(-1.);
   int firstPointInside(-1);
   for(unsigned int i=0; i<fullTrajectory_.size(); i++) {
      // LogTrace("TrackAssociator") << "Trajectory info (i,perp,r1,r2,z,z1,z2): " << i << ", " << fullTrajectory_[i].position().perp() <<
      //	", " << volume.minR() << ", " << volume.maxR() << ", " << fullTrajectory_[i].position().z() << ", " << volume.minZ() << ", " << 
      //	volume.maxZ() << ", " << closestPointOnLeft;
      dR = fullTrajectory_[i].position().perp()-volume.minR();
      dZ = fabs(fullTrajectory_[i].position().z()) - volume.minZ();
      if ( dR> 0  || dZ >0 )
	{
	   if (i>0) {
	      firstPointInside = i;
	      closestPointOnLeft = i - 1;
	   } else {
	      firstPointInside = 0;
	      closestPointOnLeft = 0;
	   }
	   break;
	}
   }
   if (closestPointOnLeft == -1) throw cms::Exception("FatalError") << "This shouls never happen - internal logic error";
   
   SteppingHelixStateInfo currentState(fullTrajectory_[closestPointOnLeft]);
   if ( currentState.position().x()*currentState.momentum().x() +
	currentState.position().y()*currentState.momentum().y() +
	currentState.position().z()*currentState.momentum().z() < 0 )
     step = -step;
   
   // propagate to the inner surface of the active volume

   if (firstPointInside != closestPointOnLeft) {
      if ( dR > 0 ) {
	 Cylinder::CylinderPointer barrel = Cylinder::build( volume.minR(), Cylinder::PositionType (0, 0, 0), Cylinder::RotationType () );
	 propagate(currentState, *barrel);
      } else {
	 Plane::PlanePointer endcap = Plane::build( Plane::PositionType (0, 0, 
									 currentState.position().z()>0?volume.minZ():-volume.minZ()), 
						    Plane::RotationType () );
	 propagate(currentState, *endcap);
      }
      if ( currentState.isValid() ) trajectory.push_back(currentState);
   } else
     LogTrace("TrackAssociator") << "Weird message\n";

   while (currentState.isValid() &&
	  currentState.position().perp()    < volume.maxR() && 
	  fabs(currentState.position().z()) < volume.maxZ() )
     {
	propagateForward(currentState,step);
	if (! currentState.isValid() ) {
	   LogTrace("TrackAssociator") << "Failed to propagate the track; moving on\n";
	   break;
	}
	// LogTrace("TrackAssociator") << "New state (perp, z): " << currentState.position().perp() << ", " << currentState.position().z();
	//if ( ( currentState.position().perp() < volume.maxR() && fabs(currentState.position().z()) < volume.maxZ() ) &&
	//     ( currentState.position().perp()-volume.minR() > 0  || fabs(currentState.position().z()) - volume.minZ() >0 ) )
	trajectory.push_back(currentState);
     }
}

void CachedTrajectory::reset_trajectory() { 
   fullTrajectory_.clear();
   ecalTrajectory_.clear();
   hcalTrajectory_.clear();
   hoTrajectory_.clear();
   preshowerTrajectory_.clear();
   wideEcalTrajectory_.clear();   
   wideHcalTrajectory_.clear();
   wideHOTrajectory_.clear();
   fullTrajectoryFilled_ = false;
}

void CachedTrajectory::findEcalTrajectory( const FiducialVolume& volume ) {
   LogTrace("TrackAssociator") << "getting trajectory in ECAL";
   getTrajectory(ecalTrajectory_, volume, 4 );
   LogTrace("TrackAssociator") << "# of points in ECAL trajectory:" << ecalTrajectory_.size();
}

void CachedTrajectory::findPreshowerTrajectory( const FiducialVolume& volume ) {
   LogTrace("TrackAssociator") << "getting trajectory in Preshower";
   getTrajectory(preshowerTrajectory_, volume, 2 );
   LogTrace("TrackAssociator") << "# of points in Preshower trajectory:" << preshowerTrajectory_.size();
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getEcalTrajectory() const{
   return ecalTrajectory_;
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getPreshowerTrajectory() const{
   return preshowerTrajectory_;
}

void CachedTrajectory::findHcalTrajectory( const FiducialVolume& volume ) {
   LogTrace("TrackAssociator") << "getting trajectory in HCAL";
   getTrajectory(hcalTrajectory_, volume, 4 ); // more steps to account for different depth
   LogTrace("TrackAssociator") << "# of points in HCAL trajectory:" << hcalTrajectory_.size();
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getHcalTrajectory() const{
   return hcalTrajectory_;
}

void CachedTrajectory::findHOTrajectory( const FiducialVolume& volume ) {
   LogTrace("TrackAssociator") << "getting trajectory in HO";
   getTrajectory(hoTrajectory_, volume, 2 );
   LogTrace("TrackAssociator") << "# of points in HO trajectory:" << hoTrajectory_.size();
}

const std::vector<SteppingHelixStateInfo>& CachedTrajectory::getHOTrajectory() const {
   return hoTrajectory_;
}
   
std::vector<GlobalPoint>*
CachedTrajectory::getWideTrajectory(const std::vector<SteppingHelixStateInfo>& states, 
                                    WideTrajectoryType wideTrajectoryType) {
   std::vector<GlobalPoint>* wideTrajectory = 0;
   switch (wideTrajectoryType) {
    case Ecal:
       LogTrace("TrackAssociator") << "Filling ellipses in Ecal trajectory";
       wideTrajectory = &wideEcalTrajectory_;
       break;
    case Hcal:
       LogTrace("TrackAssociator") << "Filling ellipses in Hcal trajectory";
      wideTrajectory = &wideHcalTrajectory_;
      break;
    case HO:
       LogTrace("TrackAssociator") << "Filling ellipses in HO trajectory";
       wideTrajectory = &wideHOTrajectory_;
       break;
   }
   if(!wideTrajectory) return 0;

   for(std::vector<SteppingHelixStateInfo>::const_iterator state= states.begin();
       state != states.end(); state++) {
      // defined a normal plane wrt the particle trajectory direction
      // let's hope that I computed the rotation matrix correctly.
      GlobalVector vector(state->momentum().unit());
      float r21 = 0;
      float r22 = vector.z()/sqrt(1-pow(vector.x(),2));
      float r23 = -vector.y()/sqrt(1-pow(vector.x(),2));
      float r31 = vector.x();
      float r32 = vector.y();
      float r33 = vector.z();
      float r11 = r22*r33-r23*r32;
      float r12 = r23*r31;
      float r13 = -r22*r31;
   
      Plane::RotationType rotation(r11, r12, r13,
                                   r21, r22, r23,
                                   r31, r32, r33);
      Plane* target = new Plane(state->position(), rotation);

      TrajectoryStateOnSurface tsos = state->getStateOnSurface(*target);

      if (!tsos.isValid()) {
         LogTrace("TrackAssociator") << "[getWideTrajectory] TSOS not valid";
         continue;
      }
      if (!tsos.hasError()) {
         LogTrace("TrackAssociator") << "[getWideTrajectory] TSOS does not have Errors";
         continue;
      }
      LocalError localErr = tsos.localError().positionError();
      localErr.scale(2); // get the 2 sigma ellipse
      float xx = localErr.xx();
      float xy = localErr.xy();
      float yy = localErr.yy();

      float denom = yy - xx;
      float phi = 0.;
      if(xy == 0 && denom==0) phi = M_PI_4;
      else phi = 0.5 * atan2(2.*xy,denom); // angle of MAJOR axis
      // Unrotate the error ellipse to get the semimajor and minor axes. Then place points on
      // the endpoints of semiminor an seminajor axes on original(rotated) error ellipse.
      LocalError rotErr = localErr.rotate(-phi); // xy covariance of rotErr should be zero
      float semi1 = sqrt(rotErr.xx());
      float semi2 = sqrt(rotErr.yy());
      
      // Just use one point if the ellipse is small
      // if(semi1 < 0.1 && semi2 < 0.1) {
      //   LogTrace("TrackAssociator") << "[getWideTrajectory] Error ellipse is small, using one trajectory point";
      //   wideTrajectory->push_back(state->position());
      //   continue;
      // }

      Local2DPoint bounds[4];
      bounds[0] = Local2DPoint(semi1*cos(phi),         semi1*sin(phi));
      bounds[1] = Local2DPoint(semi1*cos(phi+M_PI),    semi1*sin(phi+M_PI));
      phi += M_PI_2; // add pi/2 for the semi2 axis
      bounds[2] = Local2DPoint(semi2*cos(phi),         semi2*sin(phi));
      bounds[3] = Local2DPoint(semi2*cos(phi+M_PI),    semi2*sin(phi+M_PI));

      // LogTrace("TrackAssociator") << "Axes " << semi1 <<","<< semi2 <<"   phi "<< phi;
      // LogTrace("TrackAssociator") << "Local error ellipse: " << bounds[0] << bounds[1] << bounds[2] << bounds[3];

      wideTrajectory->push_back(state->position());
      for(int index=0; index<4; ++index)
         wideTrajectory->push_back(target->toGlobal(bounds[index]));

      //LogTrace("TrackAssociator") <<"Global error ellipse: (" << target->toGlobal(bounds[0]) <<","<< target->toGlobal(bounds[1])
      //         <<","<< target->toGlobal(bounds[2]) <<","<< target->toGlobal(bounds[3]) <<","<<state->position() <<")";
   }

   return wideTrajectory;
}

SteppingHelixStateInfo CachedTrajectory::getStateAtEcal()
{
   if ( ecalTrajectory_.empty() )
     return SteppingHelixStateInfo();
   else 
     return ecalTrajectory_.front();
}

SteppingHelixStateInfo CachedTrajectory::getStateAtPreshower()
{
   if ( preshowerTrajectory_.empty() )
     return SteppingHelixStateInfo();
   else 
     return preshowerTrajectory_.front();
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


SteppingHelixStateInfo CachedTrajectory::getInnerState() {
  if(fullTrajectory_.empty() )
    return SteppingHelixStateInfo();
  else
    return fullTrajectory_.front();
}


SteppingHelixStateInfo CachedTrajectory::getOuterState() {
  if(fullTrajectory_.empty() )
    return SteppingHelixStateInfo();
  else
    return fullTrajectory_.back();
}

