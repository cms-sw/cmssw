#ifndef ECAL_FENIX_PEAKFINDER_H
#define ECAL_FENIX_PEAKFINDER_H

//#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVPeakFinder.h>
#include <vector>

class  EcalVPeakFinder;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \ class EcalFenixPeakFinder
   \brief calculates the peak for Fenix strip, barrel
   *  input : 18 bits
   *  output: boolean
   *  
   *  --->gets the sample where the value is max. the  value is 1 for this max sample, 0 for the others .needs 3 samples to proceed.
   *  ----> do we really need caloVShape?
   */
//class EcalFenixPeakFinder : public EcalVPeakFinder
class EcalFenixPeakFinder {

 private:
  bool disabled;
  /* {transient=false, volatile=false}*/
  int setInput(int input);
  int process();

  int inputsAlreadyIn_;
  int buffer_[3];

 public:

/*   double getTpeak () const{return tpeak_;} */

/*   void setTpeak (double value){tpeak_=value;} */
  EcalFenixPeakFinder();
  virtual ~EcalFenixPeakFinder();
  virtual std::vector<int> process(std::vector<int>& filtout, std::vector<int> & output);
  // from CaloVShape
  virtual double operator()(double) const {return 0.;}
  virtual double derivative(double) const {return 0.;}
 };

#endif
