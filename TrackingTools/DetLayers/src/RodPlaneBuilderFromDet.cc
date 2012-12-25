#include "TrackingTools/DetLayers/interface/RodPlaneBuilderFromDet.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"

#include <algorithm>

using namespace std;

// Warning, remember to assign this pointer to a ReferenceCountingPointer!
Plane* 
RodPlaneBuilderFromDet::operator()( const vector<const Det*>& dets) const
{
  // find mean position
  typedef Surface::PositionType::BasicVectorType Vector;
  Vector posSum(0,0,0);
  for (vector<const Det*>::const_iterator i=dets.begin(); i!=dets.end(); i++) {
    posSum += (**i).surface().position().basicVector();
  }
  Surface::PositionType meanPos( posSum/float(dets.size()));
  
  // temporary plane - for the computation of bounds
  Surface::RotationType rotation = computeRotation( dets, meanPos);
  Plane tmpPlane( meanPos, rotation);
  pair<RectangularPlaneBounds,GlobalVector> bo = 
    computeBounds( dets, tmpPlane);

//   LogDebug("DetLayers") << "Creating plane at position " << meanPos 
//        << " displaced by " << bo.second ;
//   LogDebug("DetLayers") << "Bounds are (wid/len/thick) " << bo.first.width()
//        << " / " <<  bo.first.length() 
//        << " / " <<  bo.first.thickness() ;

  return new Plane( meanPos+bo.second, rotation, bo.first);
}

pair<RectangularPlaneBounds, GlobalVector>
RodPlaneBuilderFromDet::computeBounds( const vector<const Det*>& dets,
				       const Plane& plane) const
{
  // go over all corners and compute maximum deviations from mean pos.
  vector<GlobalPoint> corners;
  for (vector<const Det*>::const_iterator idet=dets.begin();
       idet != dets.end(); idet++) {

    /* ---- original implementation. Is it obsolete?
    vector<const DetUnit*> detUnits = (**idet).basicComponents();
    for (vector<const DetUnit*>::const_iterator detu=detUnits.begin();
	 detu!=detUnits.end(); detu++) {
      vector<GlobalPoint> dc = 
	BoundingBox().corners((**detu).specificSurface());
      corners.insert( corners.end(), dc.begin(), dc.end());
    }
    ---- */

    // temporary implementation (May be the final one if the GluedDet surface
    // really contains both the mono and the stereo surfaces
    vector<GlobalPoint> dc = BoundingBox().corners((**idet).specificSurface());
    corners.insert( corners.end(),dc.begin(), dc.end() );
  }
  
  float xmin(0), xmax(0), ymin(0), ymax(0), zmin(0), zmax(0);
  for (vector<GlobalPoint>::const_iterator i=corners.begin();
       i!=corners.end(); i++) {
    LocalPoint p = plane.toLocal(*i);
    if (p.x() < xmin) xmin = p.x();
    if (p.x() > xmax) xmax = p.x();
    if (p.y() < ymin) ymin = p.y();
    if (p.y() > ymax) ymax = p.y();
    if (p.z() < zmin) zmin = p.z();
    if (p.z() > zmax) zmax = p.z();
  }

  LocalVector localOffset( (xmin+xmax)/2., (ymin+ymax)/2., (zmin+zmax)/2.);
  GlobalVector offset( plane.toGlobal(localOffset));
  
  pair<RectangularPlaneBounds, GlobalVector> result(RectangularPlaneBounds((xmax-xmin)/2, (ymax-ymin)/2, (zmax-zmin)/2), offset);

  return result;
}

Surface::RotationType 
RodPlaneBuilderFromDet::
computeRotation( const vector<const Det*>& dets,
		 const Surface::PositionType& meanPos) const
{
  // choose first mono out-pointing rotation
  // the rotations of GluedDets coincide with the mono part
  // Simply take the x,y of the first Det if z points out,
  // or -x, y if it doesn't
  const Plane& plane =
    dynamic_cast<const Plane&>(dets.front()->surface());
  //GlobalVector n = plane.normalVector();

  GlobalVector xAxis;
  GlobalVector yAxis;
  GlobalVector planeYAxis = plane.toGlobal( LocalVector( 0, 1, 0));
  if (planeYAxis.z() < 0) yAxis = -planeYAxis;
  else                    yAxis =  planeYAxis;

  GlobalVector planeXAxis = plane.toGlobal( LocalVector( 1, 0, 0));
  GlobalVector n = planeXAxis.cross( planeYAxis);

  if (n.x() * meanPos.x() + n.y() * meanPos.y() > 0) {
    xAxis = planeXAxis;
  }
  else {
    xAxis = -planeXAxis;
  }

//   LogDebug("DetLayers") << "Creating rotation with x,y axis " 
//        << xAxis << ", " << yAxis ;

  return Surface::RotationType( xAxis, yAxis);
}
