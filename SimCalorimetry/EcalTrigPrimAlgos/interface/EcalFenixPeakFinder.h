#ifndef ECAL_FENIX_PEAKFINDER_H
#define ECAL_FENIX_PEAKFINDER_H

#include <vector>

  /** 
   \ class EcalFenixPeakFinder
   \brief calculates the peak for Fenix strip, barrel
   *  input : 18 bits
   *  output: boolean
   *  
   *  --->gets the sample where the value is max. the  value is 1 for this max sample, 0 for the others .needs 3 samples to proceed.
   *  ----> do we really need caloVShape?
   */

class EcalFenixPeakFinder {

 private:
  bool disabled;
  int setInput(int input);
  int process();

  int inputsAlreadyIn_;
  int buffer_[3];

 public:

  EcalFenixPeakFinder();
  virtual ~EcalFenixPeakFinder();
  virtual std::vector<int> process(std::vector<int>& filtout, std::vector<int> & output);
  // from CaloVShape
  //  virtual double operator()(double) const {return 0.;}
  //  virtual double derivative(double) const {return 0.;}
 };

#endif
