#ifndef EcalShape_h
#define EcalShape_h
#include<vector>
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
  
  /**
  
     \class EcalShape
  
     \brief  shaper for Ecal
     
  */
  
namespace cms {

  class EcalShape : public CaloVShape
  {
  public:
    
    EcalShape()
      {setTpeak(47.6683);}
    
    EcalShape(const EcalShape&d):
      CaloVShape(d),tconv(d.tconv),nbin(d.nbin),nt(d.nt),ntd(d.ntd)
      {setTpeak(47.6683);}
  
    ~EcalShape(){}
    
    double operator () (double time_) const;
    void display () const {}
    double derivative (double time_) const;
    double getTpeak () const;
    
    void computeShape();
  
   private:
    
    int tconv;
    int nbin;
    std::vector<float> nt;
    std::vector<float> ntd;
    
  };
  
}

#endif
  
