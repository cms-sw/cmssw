#ifndef ECAL_FENIX_AMPLITUDE_FILTER_H
#define ECAL_FENIX_AMPLITUDE_FILTER_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAmplitudeFilter.h>
#include <stdio.h>
#include <iostream>

class EcalVAmplitudeFilter;
class EcalTPParameters;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \ class EcalFenixAmplitudeFilter
   \brief calculates .... for Fenix strip, barrel
   *  input: 18 bits
   *  output: 18 bits
   *  
   */
class EcalFenixAmplitudeFilter : public EcalVAmplitudeFilter {


 private:
  const EcalTPParameters *ecaltpp_ ;
  int inputsAlreadyIn_;
  int buffer_[5];
  int weights_[5];
  int shift_;
  int setInput(int input);
  int process();
  


 public:
  EcalFenixAmplitudeFilter(const EcalTPParameters * db);
  virtual ~EcalFenixAmplitudeFilter();
  virtual void process(std::vector<int> & addout, std::vector<int> & output);
  void setParameters(int SM, int towerInSM, int stripInTower);

  
};

#endif
