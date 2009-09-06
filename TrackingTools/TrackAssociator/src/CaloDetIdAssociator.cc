#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
bool CaloDetIdAssociator::crossedElement(const GlobalPoint& point1,
					 const GlobalPoint& point2,
					 const DetId& id,
					 const double toleranceInSigmas,
					 const SteppingHelixStateInfo* initialState
					 ) const
{
   // Define plane normal to the trajectory direction at the first point
   GlobalVector vector = (point2-point1).unit();
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
   Plane::PlanePointer plane = Plane::build(point1, rotation);
   double absoluteTolerance = -1;
   if ( toleranceInSigmas>0 && initialState ){
      TrajectoryStateOnSurface tsos = initialState->getStateOnSurface(*plane);
      if ( tsos.isValid() and tsos.hasError()) {
	 LocalError localErr = tsos.localError().positionError();
	 localErr.scale(toleranceInSigmas); 
	 float xx = localErr.xx();
	 float xy = localErr.xy();
	 float yy = localErr.yy();

	 float denom = yy - xx;
	 float phi = 0., phi_temp=0.;
	 if(xy == 0 && denom==0) phi = M_PI_4;
	 else phi = 0.5 * atan2(2.*xy,denom); // angle of MAJOR axis
	 phi_temp = phi;
	 // Unrotate the error ellipse to get the semimajor and minor axes. Then place points on
	 // the endpoints of semiminor an seminajor axes on original(rotated) error ellipse.
	 LocalError rotErr = localErr.rotate(-phi); // xy covariance of rotErr should be zero
	 float semi1 = sqrt(rotErr.xx());
	 float semi2 = sqrt(rotErr.yy());
	 absoluteTolerance = std::max(semi1,semi2);
      }
   }
   
   // distance between the points.
   double trajectorySegmentLength = vector.mag();

   // we need to find the angle that covers all points. 
   // if it's bigger than 180 degree, we are inside
   // otherwise we are outside, i.e. the volume is not crossed
   bool allBehind = true;
   bool allTooFar = true;
   const std::vector<GlobalPoint>& points = getDetIdPoints(id);
   std::vector<GlobalPoint>::const_iterator p = points.begin();
   if ( p == points.end() ) {
      edm::LogWarning("TrackAssociator") << "calo geometry for element " << id.rawId() << "is empty. Ignored"; 
      return false; 
   }
   LocalPoint localPoint = plane->toLocal(*p);
   double minPhi = localPoint.phi();
   double maxPhi = localPoint.phi();
   if ( localPoint.z() < 0 ) 
     allTooFar = false;
   else {  
      allBehind = false;
      if ( localPoint.z() < trajectorySegmentLength )  allTooFar = false;
   }
   ++p;
   for (; p!=points.end(); ++p){
      localPoint = plane->toLocal(*p);
      double localPhi = localPoint.phi();
      if ( localPoint.z() < 0 ) 
	allTooFar = false;
      else {  
	 allBehind = false;
	 if ( localPoint.z() < trajectorySegmentLength )  allTooFar = false;
      }
      if ( localPhi >= minPhi && localPhi <= maxPhi ) continue;
      if ( localPhi+2*M_PI >= minPhi && localPhi+2*M_PI <= maxPhi ) continue;
      if ( localPhi-2*M_PI >= minPhi && localPhi-2*M_PI <= maxPhi ) continue;
      // find the closest limit
      if ( localPhi > maxPhi ){
	 double delta1 = fabs(localPhi-maxPhi);
	 double delta2 = fabs(localPhi-2*M_PI-minPhi);
	 if ( delta1 < delta2 )
	   maxPhi = localPhi;
	 else
	   minPhi = localPhi-2*M_PI;
	 continue;
      }
      if ( localPhi < minPhi ){
	 double delta1 = fabs(localPhi-minPhi);
	 double delta2 = fabs(localPhi+2*M_PI-maxPhi);
	 if ( delta1 < delta2 )
	   minPhi = localPhi;
	 else
	   maxPhi = localPhi+2*M_PI;
	 continue;
      }
      cms::Exception("FatalError") << "Algorithm logic error - this should never happen. Problems with trajectory-volume matching.";
   }
   if ( allBehind ) return false;
   if ( allTooFar ) return false;
   if ( fabs(maxPhi-minPhi)>M_PI ) return true;
   
   // now if the tolerance is positive, check how far we are 
   // from the closest line segment
   if (absoluteTolerance < 0 ) return false;
   double distanceToClosestLineSegment = 1e9;
   for ( unsigned int i=0; i+1 < points.size(); ++i )
     for ( unsigned int j=i+1; j < points.size(); ++j )
       {
	  LocalPoint p1(plane->toLocal(points[i]));
	  LocalPoint p2(plane->toLocal(points[j]));
	  // now we deal with high school level math to get
	  // the triangle paramaters
	  double side1squared = p1.perp2();
	  double side2squared = p2.perp2();
	  double side3squared = (p2.x()-p1.x())*(p2.x()-p1.x()) + (p2.y()-p1.y())*(p2.y()-p1.y());
	  double area = fabs(p1.x()*p2.y()-p2.x()*p1.y())/2;
	  // all triangle angles must be smaller than 90 degree
	  // otherwise the projection is out of the line segment
	  if ( side1squared + side2squared > side3squared &&
	       side2squared + side3squared > side1squared &&
	       side1squared + side3squared > side1squared )
	    {
	       double h(2*area/sqrt(side3squared));
	       if ( h < distanceToClosestLineSegment ) distanceToClosestLineSegment = h;
	    }
	  else
	    {
	       if ( sqrt(side1squared) < distanceToClosestLineSegment ) distanceToClosestLineSegment = sqrt(side1squared);
	       if ( sqrt(side2squared) < distanceToClosestLineSegment ) distanceToClosestLineSegment = sqrt(side2squared);
	    }
       }
   if ( distanceToClosestLineSegment < absoluteTolerance ) return true;
   return false;
}
