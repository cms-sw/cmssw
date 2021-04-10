#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXPEAKFINDERPHASE1_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXPEAKFINDERPHASE1_H

#include <vector>

/**
 \ class EcalFenixPeakFinderPhase1
 \brief calculates the peak for Fenix strip, barrel
 *  input : 18 bits
 *  output: boolean
 *
 *  --->gets the sample where the value is max. the  value is 1 for this max
 sample, 0 for the others .needs 3 samples to proceed.
 *  ----> do we really need caloVShape?
 */

class EcalFenixPeakFinderPhase1{
private:
  bool disabled;
  int setInput(int input);
  int process();

  int inputsAlreadyIn_;
  int buffer_[3];

public:
  EcalFenixPeakFinderPhase1();
  virtual ~EcalFenixPeakFinderPhase1();
  virtual std::vector<int> process(std::vector<int> &filtout, std::vector<int> &output);
  // from CaloVShape
  //  virtual double operator()(double) const {return 0.;}
  //  virtual double derivative(double) const {return 0.;}
};

#endif
