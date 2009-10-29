#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
bool CaloDetIdAssociator::crossedElement(const GlobalPoint& point1,
					 const GlobalPoint& point2,
					 const DetId& id,
					 const double toleranceInSigmas,
					 const SteppingHelixStateInfo* initialState
					 ) const
{
   const std::pair<const_iterator,const_iterator>& points = getDetIdPoints(id);
   // fast check
   bool xLess(false), xIn(false), xMore(false);
   bool yLess(false), yIn(false), yMore(false);
   bool zLess(false), zIn(false), zMore(false);
   double xMin(point1.x()), xMax(point2.x());
   double yMin(point1.y()), yMax(point2.y());
   double zMin(point1.z()), zMax(point2.z());
   if ( xMin>xMax ) std::swap(xMin,xMax);
   if ( yMin>yMax ) std::swap(yMin,yMax);
   if ( zMin>zMax ) std::swap(zMin,zMax);
   for ( std::vector<GlobalPoint>::const_iterator it = points.first;
	 it != points.second; ++it ){
     if ( it->x()<xMin ){
       xLess = true;
     } else {
       if ( it->x()>xMax )
	 xMore = true;
       else
	 xIn = true;
     }
     if ( it->y()<yMin ){
       yLess = true;
     } else {
       if ( it->y()>yMax )
	 yMore = true;
       else
	 yIn = true;
     }
     if ( it->z()<zMin ){
       zLess = true;
     } else {
       if ( it->z()>zMax )
	 zMore = true;
       else
	 zIn = true;
     }
   }
   if ( ( (xLess && !xIn && !xMore) || (!xLess && !xIn && xMore) ) ||
	( (yLess && !yIn && !yMore) || (!yLess && !yIn && yMore) ) ||
	( (zLess && !zIn && !zMore) || (!zLess && !zIn && zMore) ) ) return false;

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
   double trajectorySegmentLength = (point2-point1).mag();

   // we need to find the angle that covers all points. 
   // if it's bigger than 180 degree, we are inside
   // otherwise we are outside, i.e. the volume is not crossed
   bool allBehind = true;
   bool allTooFar = true;
   std::vector<GlobalPoint>::const_iterator p = points.first;
   if ( p == points.second ) {
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
   for (; p!=points.second; ++p){
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
   std::vector<GlobalPoint>::const_iterator i,j;
   for ( i = points.first; i != points.second; ++i )
     for ( j = i+1; j != points.second; ++j )
       {
	  LocalPoint p1(plane->toLocal(*i));
	  LocalPoint p2(plane->toLocal(*j));
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

void CaloDetIdAssociator::setGeometry(const DetIdAssociatorRecord& iRecord)
{
  edm::ESHandle<CaloGeometry> geometryH;
  iRecord.getRecord<CaloGeometryRecord>().get(geometryH);
  setGeometry(geometryH.product());
}

void CaloDetIdAssociator::check_setup() const
{
  DetIdAssociator::check_setup();
  if (geometry_==0) throw cms::Exception("CaloGeometry is not set");
}
   
GlobalPoint CaloDetIdAssociator::getPosition(const DetId& id) const {
  return geometry_->getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
}
   
std::set<DetId> CaloDetIdAssociator::getASetOfValidDetIds() const 
{
  std::set<DetId> setOfValidIds;
  const std::vector<DetId>& vectOfValidIds = geometry_->getValidDetIds(DetId::Calo, 1);
  for(std::vector<DetId>::const_iterator it = vectOfValidIds.begin(); it != vectOfValidIds.end(); ++it)
    setOfValidIds.insert(*it);
  
  return setOfValidIds;
}
   
std::pair<DetIdAssociator::const_iterator, DetIdAssociator::const_iterator> 
CaloDetIdAssociator::getDetIdPoints(const DetId& id) const 
{
  const CaloSubdetectorGeometry* subDetGeom = geometry_->getSubdetectorGeometry(id);
  if(! subDetGeom){
    LogDebug("TrackAssociator") << "Cannot find sub-detector geometry for " << id.rawId() <<"\n";
    return std::pair<const_iterator,const_iterator>(dummy_.end(),dummy_.end());
  }
  const CaloCellGeometry* cellGeom = subDetGeom->getGeometry(id);
  if(! cellGeom) {
    LogDebug("TrackAssociator") << "Cannot find CaloCell geometry for " << id.rawId() <<"\n";
    return std::pair<const_iterator,const_iterator>(dummy_.end(),dummy_.end());
  } 
  const CaloCellGeometry::CornersVec& cor (cellGeom->getCorners() ) ; 
  return std::pair<const_iterator,const_iterator>( cor.begin(), cor.end() ) ;
}
