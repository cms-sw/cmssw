#ifndef KinematicVertexDistance_H
#define KinematicVertexDistance_H

/** 
 *  Abstact class which defines a distance and compatibility between 
 * a  SimVertex descendent and a kinematicVertex.
 */

#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "TrackerReco/TkEvent/interface/TkSimVertex.h"

class Measurement1D;

class KinematicVertexDistance{

public:

  virtual ~KinematicVertexDistance() {}

  virtual Measurement1D distance(const RefCountedKinematicVertex, const TkSimVertex &) const = 0;

  virtual float compatibility (const RefCountedKinematicVertex, const TkSimVertex &) const = 0;

  virtual KinematicVertexDistance * clone() const = 0;
};
#endif
