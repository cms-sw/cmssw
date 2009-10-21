#ifndef EcalSimAlgos_EcalShape_h
#define EcalSimAlgos_EcalShape_h

#include<vector>
#include<stdexcept>
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
  
/**
   \class EcalShape
   \brief  shaper for Ecal
*/
class EcalShape : public CaloVShape
{
public:
  
  /// ctor
  EcalShape(double timePhase);
  /// dtor
  ~EcalShape(){}
  
  double operator () (double time_) const;
  double derivative (double time_) const;
  
  double computeTimeOfMaximum() const;
  double computeT0() const;
  double computeRisingTime() const;

  void   load(int xtal_, int SuperModule_); //modif Alex 20/07/07
  const std::vector<double>& getTimeTable() const;
  const std::vector<double>& getDerivTable() const;

 private:

  int nsamp;
  int tconv;
  int nbin;
  std::vector<double> nt;
  std::vector<double> ntd;

  double threshold;
  int binstart;

};
  


#endif
  
