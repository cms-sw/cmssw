#ifndef EcalSimAlgos_ESShape_h
#define EcalSimAlgos_ESShape_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

class ESShape : public CaloVShape 
{
  
 public:
  
  ESShape()
    {
      setTpeak(20.0);
    }     
  
  ~ESShape(){}
  
  double operator () (double time_) const;
  void display () const {}
  double derivative (double time_) const;
  double getTpeak () const;      

  static const double A = 6. ;
  static const double Qcf = 4./350. ;
  static const double omegac = 2./25. ;
  static const double norm = 0.11136 ;

};

#endif
