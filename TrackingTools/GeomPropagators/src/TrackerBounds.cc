#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"


//Ported from ORCA
//  $Date: 2006/04/21 14:30:42 $
//  $Revision: 1.1 $

void TrackerBounds::initialize() 
{
  const float epsilon = 0.001; // should not matter at all

  Surface::RotationType rot; // unit rotation matrix

  theCylinder = new BoundCylinder( Surface::PositionType(0,0,0), rot, 
				   SimpleCylinderBounds( radius()-epsilon, 
							 radius()+epsilon, 
							 -halfLength(), 
							 halfLength()));
  theNegativeDisk = 
    new BoundDisk( Surface::PositionType( 0, 0, -halfLength()), rot, 
		   SimpleDiskBounds( 0, radius(), -epsilon, epsilon));

  thePositiveDisk = 
    new BoundDisk( Surface::PositionType( 0, 0, halfLength()), rot, 
		   SimpleDiskBounds( 0, radius(), -epsilon, epsilon));


  theInit = true;
}

bool TrackerBounds::isInside(const GlobalPoint &point){
  return (point.perp() <= radius() &&
	  fabs(point.z()) <= halfLength());
}


// static initializers

ReferenceCountingPointer<BoundCylinder>  TrackerBounds::theCylinder = 0;
ReferenceCountingPointer<BoundDisk>      TrackerBounds::theNegativeDisk = 0;
ReferenceCountingPointer<BoundDisk>      TrackerBounds::thePositiveDisk = 0;

bool TrackerBounds::theInit = false;


