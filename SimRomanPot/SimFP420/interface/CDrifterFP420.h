#ifndef CDrifterFP420_h
#define CDrifterFP420_h

#include "SimRomanPot/SimFP420/interface/EnergySegmentFP420.h"
#include "SimRomanPot/SimFP420/interface/AmplitudeSegmentFP420.h"
//#include "ClassReuse/GeomVector/interface/LocalVector.h"
#include "G4StepPoint.hh"


#include<vector>

class CDrifterFP420{
 public:  
  typedef std::vector <AmplitudeSegmentFP420> collection_type;
  typedef std::vector <EnergySegmentFP420> ionization_type;

  virtual ~CDrifterFP420() { }

  virtual collection_type drift (const ionization_type, const G4ThreeVector&, const int&) = 0;
//  virtual collection_type drift (const ionization_type, const G4ThreeVector&) = 0;

};

#endif

