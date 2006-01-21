#ifndef HcalSimAlgos_HFShape_h
#define HcalSimAlgos_HFShape_h
#include<vector>
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
  
/**
  
   \class HFShape
  
   \brief  shaper for HF
     
*/
  

class HFShape : public CaloVShape
{
public:
  
  HFShape()
  {   setTpeak(2.0); 
      computeShapeHF();
  }
  
  HFShape(const HFShape&d):
    CaloVShape(d),nbin(d.nbin),nt(d.nt)
    {setTpeak(2.0);}

  ~HFShape(){}
  
  double operator () (double time_) const;
  void display () const {}
  double derivative (double time_) const;
  double getTpeak () const;

  void computeShapeHF();

 private:
  
  int nbin;
  std::vector<float> nt;
  
};

#endif
  
