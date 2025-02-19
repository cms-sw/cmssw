#ifndef PropagationTestGenerator_H_
#define PropagationTestGenerator_H_

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/** \class PropagationTestGenerator
 *  Base class for generation of test trajectories (currently 
 *  HelixPropagationTestGenerator and StraightLinePropagationTestGenerator).
 */

class PropagationTestGenerator {

protected:
  typedef long double ExtendedDouble;
  typedef Basic3DVector<ExtendedDouble> VectorTypeExtended;

public:
  //
  // Constructor/Destructor
  //
  PropagationTestGenerator();
  virtual ~PropagationTestGenerator() {}

  /// Range of azimuthal angles for trajectory generation
  virtual void setRangePhi(const float min, const float max);
  /// Range of pseudo-rapidities for trajectory generation
  virtual void setRangeEta(const float, const float);
  /// Gaussian smearing of vertex
  virtual void setVertexSmearing(const float, const float, const float);

  /// Returns curvature (transversal!).
  virtual ExtendedDouble transverseCurvature() const;
  /// Returns charge.
  virtual int charge() const;

  /** Steps forward along the trajectory by a pathlength between
   *  -(argument) and 0. Returns total pathlength from
   *  starting point after the step.
   */
  virtual ExtendedDouble randomStepForward(const float);
  /** Steps backward along the trajectory by a pathlength between
   *  -(argument) and 0. Returns total pathlength from
   *  starting point after the step.
   */
  virtual ExtendedDouble randomStepBackward(const float);

  /// Returns current position (start or after last step).
  virtual GlobalPoint position() const;
  /// Returns current direction (start or after last step).
  virtual GlobalVector momentum() const;

  /// Generates random start values for a new trajectory
  virtual void generateStartValues() =0;
  /// Sets start values to current position / momentum
  virtual void setStartToCurrent() =0;
  /** Sets start values to current position / 
   * inverted momentum vector / inverted charge */
  virtual void setStartOppositeToCurrent() =0;

protected:
  /** Step in forward or backward direction by 
   *  pathlength = argument.
   *  Return value = step size.
   */
  virtual ExtendedDouble bidirectionalStep (const ExtendedDouble) =0;

protected:
  bool initialised;

  float phiMin;
  float phiMax;
  float etaMin;
  float etaMax;
  float posVx;
  float posVy;
  float posVz;
  float sigVx;
  float sigVy;
  float sigVz;

  bool useLogStep;

  VectorTypeExtended startPosition;
  VectorTypeExtended startDirection;
  ExtendedDouble theCurvature;
  int theCharge;

  ExtendedDouble sTotal;
  VectorTypeExtended currentPosition;
  VectorTypeExtended currentDirection;
};

#endif
