#ifndef CDividerFP420_h
#define CDividerFP420_h

#include "SimRomanPot/SimFP420/interface/EnergySegmentFP420.h"
//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"


#include <vector>

class CDividerFP420{
 public:
  
  typedef std::vector< EnergySegmentFP420 > ionization_type;
  
  virtual ~CDividerFP420() { }
  //  virtual ionization_type divide(const PSimHit, const StripDet& det) = 0;
  virtual ionization_type divide(const PSimHit&, const double&) = 0;

};


#endif
