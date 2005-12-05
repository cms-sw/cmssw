///////////////////////////////////////////////////////////////////////////////
// File: EcalNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for ECal
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalNumberingScheme_h
#define EcalNumberingScheme_h

#include "SimG4CMS/Calo/interface/CaloNumberingScheme.h"
#include <boost/cstdint.hpp>
#include <map>

class EcalNumberingScheme : public CaloNumberingScheme {

public:
  EcalNumberingScheme(int);
  virtual ~EcalNumberingScheme();
  virtual uint32_t getUnitID(const G4Step* aStep) const = 0;

  // additional tools
  typedef std::map<uint32_t,float> MapType;
  uint32_t getUnitWithMaxEnergy(MapType& themap);
  virtual float energyInMatrix(int nCellInEta, int nCellInPhi, 
			       int centralEta, int centralPhi, int centralZ,
			       MapType& themap) {return 0;}
};

#endif
