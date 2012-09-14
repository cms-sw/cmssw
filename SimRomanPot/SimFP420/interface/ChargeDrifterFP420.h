#ifndef ChargeDrifterFP420_h
#define ChargeDrifterFP420_h

#include "SimRomanPot/SimFP420/interface/CDrifterFP420.h"
#include "SimRomanPot/SimFP420/interface/EnergySegmentFP420.h"

class ChargeDrifterFP420 : public CDrifterFP420{
 public:
  ChargeDrifterFP420(double,double,double,double,double,double,double,double,double,int);
  CDrifterFP420::collection_type drift (const CDrifterFP420::ionization_type, const G4ThreeVector&, const int&);

 private:
  
  AmplitudeSegmentFP420 drift(const EnergySegmentFP420&, const G4ThreeVector&, const int&);
  
  double modulePath;
  double constTe;
  double constDe;
  double temperature;
  double startT0;
  double depV;
  double appV;
  double ldriftcurrX;
  double ldriftcurrY;
  int verbo;
  
};


#endif

