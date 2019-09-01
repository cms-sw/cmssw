#include "TrackingTools/GeomPropagators/interface/TrackerBounds.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

//Ported from ORCA

static const float epsilon = 0.001;  // should not matter at all

static Cylinder* initCylinder() {
  Surface::RotationType rot;  // unit rotation matrix
  auto cb = new SimpleCylinderBounds(TrackerBounds::radius() - epsilon,
                                     TrackerBounds::radius() + epsilon,
                                     -TrackerBounds::halfLength(),
                                     TrackerBounds::halfLength());
  return new Cylinder(Cylinder::computeRadius(*cb), Surface::PositionType(0, 0, 0), rot, cb);
}

static Disk* initNegative() {
  Surface::RotationType rot;  // unit rotation matrix

  return new Disk(Surface::PositionType(0, 0, -TrackerBounds::halfLength()),
                  rot,
                  new SimpleDiskBounds(0, TrackerBounds::radius(), -epsilon, epsilon));
}

static Disk* initPositive() {
  Surface::RotationType rot;  // unit rotation matrix

  return new Disk(Surface::PositionType(0, 0, TrackerBounds::halfLength()),
                  rot,
                  new SimpleDiskBounds(0, TrackerBounds::radius(), -epsilon, epsilon));
}

bool TrackerBounds::isInside(const GlobalPoint& point) {
  return (point.perp() <= TrackerBounds::radius() && fabs(point.z()) <= TrackerBounds::halfLength());
}

// static initializers

const ReferenceCountingPointer<BoundCylinder> TrackerBounds::theCylinder = initCylinder();
const ReferenceCountingPointer<BoundDisk> TrackerBounds::theNegativeDisk = initNegative();
const ReferenceCountingPointer<BoundDisk> TrackerBounds::thePositiveDisk = initPositive();
