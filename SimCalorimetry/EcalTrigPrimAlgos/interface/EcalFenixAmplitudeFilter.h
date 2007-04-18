#ifndef ECAL_FENIX_AMPLITUDE_FILTER_H
#define ECAL_FENIX_AMPLITUDE_FILTER_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAmplitudeFilter.h>
#include <stdio.h>
#include <iostream>

class EcalVAmplitudeFilter;
class DBInterface ;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \ class EcalFenixAmplitudeFilter
   \brief calculates .... for Fenix strip, barrel
   *  input: 18 bits
   *  output: 18 bits
   *  
   *  
   *  One should look at candidate implementations:
   *  EdrAnalyserWithChi2, EdrAnalyser in Detailed. They implement the CaloVAnalyser interface.
   *  
   *  And at EdShape in Detailed which implements CaloVShape
   *  
   *  The inheritance from CaloVShape is just a guess! It should be reevaluated after having studied EdrAnalyser.
   *  
   *  --->maybe not inheritance from caloVShape but more from CaloVAnalyser...or something else . do we really need caloVAnalyser?
   *  ----> computes the weighted sum ---> needs some external info (weights) . We need one set of weights per strip but it seems that weights can be the same for all the strips in the future.
   */
class EcalFenixAmplitudeFilter : public EcalVAmplitudeFilter {


 private:
  DBInterface * db_ ;
  int inputsAlreadyIn_;
  int buffer_[5];
  int weights_[5];
  int shift_;
  int setInput(int input);
  int process();
  


 public:
  EcalFenixAmplitudeFilter(DBInterface * db);
  virtual ~EcalFenixAmplitudeFilter();
  virtual std::vector<int> process(std::vector<int>);
  void setParameters(int SM, int towerInSM, int stripInTower);

  //  virtual EvalAmplitude evalAmplitude(const CaloTimeSample&) const {return EvalAmplitude();}
  
};

#endif
