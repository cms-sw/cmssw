#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

//Ported from ORCA
//  $Date: 2007/03/07 16:28:42 $
//  $Revision: 1.3 $

void TrackerBounds::initialize() 
{
  const float epsilon = 0.001; // should not matter at all

  Surface::RotationType rot; // unit rotation matrix
  auto cb = new SimpleCylinderBounds( radius()-epsilon, 
				      radius()+epsilon, 
				      -halfLength(),  halfLength()
                                     );

 theCylinder = new Cylinder(Cylinder::computeRadius(*cb), Surface::PositionType(0,0,0), rot, cb);


  theNegativeDisk = 
    new Disk( Surface::PositionType( 0, 0, -halfLength()), rot, 
		   new SimpleDiskBounds( 0, radius(), -epsilon, epsilon));

  thePositiveDisk = 
    new Disk( Surface::PositionType( 0, 0, halfLength()), rot, 
		   new SimpleDiskBounds( 0, radius(), -epsilon, epsilon));


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


