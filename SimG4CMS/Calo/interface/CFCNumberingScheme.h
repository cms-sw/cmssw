#ifndef SimG4CMS_CFCNumberingScheme_h
#define SimG4CMS_CFCNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: CFCNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for CFC
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"
#include <boost/cstdint.hpp>
#include <vector>

class CFCNumberingScheme : public CaloNumberingScheme {

public:
  CFCNumberingScheme(std::vector<double> rv, std::vector<double> xv, 
		     std::vector<double> yv);
  virtual ~CFCNumberingScheme();
  virtual uint32_t getUnitID(G4ThreeVector point, int iz, int mod,
                             int fibType, int depth=0);

private:
  CFCNumberingScheme();

  std::vector<double>    rTable, xCellSize, yCellSize;
  std::vector<int>       nMaxX;
};

#endif
