#ifndef EcalSimAlgos_ESShape_h
#define EcalSimAlgos_ESShape_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

class ESShape : public CaloVShape 
{
 // Preshower pulse shape
 // Gain = 0 : old shape used in ORCA
 // Gain = 1 : shape for low gain for data taking
 // Gain = 2 : shape for high gain for calibration
 // Preshower three time samples happen at -5, 20 and 45 ns 
  
 public:
  
  ESShape(int Gain);
  ~ESShape(){}
  
  double operator () (double time_) const;
  void display () const {}
  double derivative (double time_) const;
  double getTpeak () const;      

 private:

  int theGain;
  double A;
  double Qcf;
  double omegac;
  double norm;

};

#endif
