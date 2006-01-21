#ifndef HcalSimAlgos_HcalShape_h
#define HcalSimAlgos_HcalShape_h
#include<vector>
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
  
/**

   \class HcalShape

   \brief  shaper for Hcal (not for HF)
   
*/

class HcalShape : public CaloVShape
{
public:
  
  HcalShape()
  {
     setTpeak(32.0);
     computeShape();
  }
  
  HcalShape(const HcalShape&d):
    CaloVShape(d),nbin(d.nbin),nt(d.nt)
    {setTpeak(32.0);}

  ~HcalShape(){}
  
  double operator () (double time_) const;
  void display () const {}
  double derivative (double time_) const;
  double getTpeak () const;

  void computeShape();

 private:
  
  int nbin;
  std::vector<float> nt;
  
};

#endif
  
  
