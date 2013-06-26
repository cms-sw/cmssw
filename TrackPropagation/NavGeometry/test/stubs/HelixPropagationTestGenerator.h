#ifndef HelixPropagationTestGenerator_H_
#define HelixPropagationTestGenerator_H_

#include "TrackPropagation/NavGeometry/test/stubs/PropagationTestGenerator.h"

class MagneticField;

/** \class HelixPropagationTestGenerator
 *  Generates random helices and points / directions at successive random
 *  forward / backward steps.
 */

class HelixPropagationTestGenerator : public PropagationTestGenerator {
public:
  //
  // Constructor/Destructor
  //
  HelixPropagationTestGenerator(const MagneticField* field);
  ~HelixPropagationTestGenerator() {}

  /// Range of charges for helix generation (+/-1)
  void setRangeCharge(const float, const float);
  /// Range of azimuthal angles for helix generation
  void setRangePt(const float, const float);
  /// Generates a new helix.
  virtual void generateStartValues ();
  /// Returns position of center (z acc. to starting point).
  GlobalPoint center() const;

  /** Sets start values according to position /
   * momentum vector. */
  void setStart(const GlobalPoint&, const GlobalVector&);
  /// Sets start values to current position / momentum
  virtual void setStartToCurrent();
  /** Sets start values to current position / 
   * inverted momentum vector / inverted charge */
  virtual void setStartOppositeToCurrent();

private:
  /** Step in forward or backward direction by 
   *  pathlength = argument. 
   *  Return value = step size.
   */
  virtual ExtendedDouble bidirectionalStep (const ExtendedDouble);

private:
  float qMin;
  float qMax;
  float ptMin;
  float ptMax;

  bool useLogPt;

  VectorTypeExtended theCenter;
  ExtendedDouble startPhiHelix;

  const MagneticField* theField;
};

#endif
