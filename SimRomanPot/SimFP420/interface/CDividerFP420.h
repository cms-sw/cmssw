#ifndef CDividerFP420_h
#define CDividerFP420_h

#include "SimRomanPot/SimFP420/interface/EnergySegmentFP420.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"

using namespace std;
#include <vector>

class CDividerFP420{
 public:
  
  typedef vector< EnergySegmentFP420 > ionization_type;
  
  virtual ~CDividerFP420() { }
  //  virtual ionization_type divide(const FP420G4Hit, const StripDet& det) = 0;
  virtual ionization_type divide(const FP420G4Hit&, const double&) = 0;

};


#endif
