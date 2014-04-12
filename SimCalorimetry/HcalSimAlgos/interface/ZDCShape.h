#ifndef HcalSimAlgos_ZDCShape_h
#define HcalSimAlgos_ZDCShape_h
#include<vector>
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
  
/**
  
   \class ZDCShape
  
   \brief  shaper for ZDC
     
*/
  

class ZDCShape : public CaloVShape
{
public:
  
  ZDCShape();
  ZDCShape(const ZDCShape&d);

  virtual ~ZDCShape(){}
  
  virtual double operator () (double time) const;
  virtual double       timeToRise()         const  ;


 private:
  void computeShapeZDC();
  
  int nbin_;
  std::vector<float> nt_;
  
};

#endif
  
