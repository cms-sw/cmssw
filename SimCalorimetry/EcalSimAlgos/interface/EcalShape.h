#ifndef EcalSimAlgos_EcalShape_h
#define EcalSimAlgos_EcalShape_h

#include<vector>
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
  
/**

   \class EcalShape

   \brief  shaper for Ecal
   
*/
  

class EcalShape : public CaloVShape
{
public:
  
  EcalShape();

  ~EcalShape(){}
  
  double operator () (double time_) const;
  void display () const {}
  double derivative (double time_) const;
  double getTpeak () const;
  
 private:
  
  int tconv;
  int nbin;
  std::vector<float> nt;
  std::vector<float> ntd;
    
};
  


#endif
  
