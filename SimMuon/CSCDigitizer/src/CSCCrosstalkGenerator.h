#ifndef MU_END_CROSSTALK_GENERATOR_H
#define MU_END_CROSSTALK_GENERATOR_H

class CSCAnalogSignal;

/** \class CSCCrosstalkGenerator
 *
 * Cross-talk generator for digitization simulation of Endcap Muon CSCs.
 * We model crosstalk by making the signal on a neighboring
 *    strip or wire proportional to the slope of the original
 *    signal.  The constant should be chosen to give the appropriate
 *    level of crosstalk, maybe 10% of the signal. The user is responsible
 *    for subtracting the crosstalk from the input signal,
 *    and adding the crosstalk signal to the neighbors.
 *
 *  \author Rick Wilkinson,
 */

class CSCCrosstalkGenerator {
public:
  CSCCrosstalkGenerator() : theCrosstalk(0), theDelay(0), theResistiveFraction(0.){};

  void setParameters(float crosstalk, float delay, float resistiveFraction) {
    theCrosstalk = crosstalk;
    theDelay = delay;
    theResistiveFraction = resistiveFraction;
  }

  CSCAnalogSignal getCrosstalk(const CSCAnalogSignal &inputSignal) const;

  /// analyzes the ratio between two signals.
  float ratio(const CSCAnalogSignal &crosstalkSignal, const CSCAnalogSignal &signal) const;

private:
  float theCrosstalk;
  float theDelay;
  // what fraction of the neighboring signal goes unaltered onto this element
  float theResistiveFraction;
};

#endif
