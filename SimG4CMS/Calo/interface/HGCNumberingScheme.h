#ifndef SimG4CMS_HGCNumberingScheme_h
#define SimG4CMS_HGCNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for HGC
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"
#include <boost/cstdint.hpp>
#include <vector>

class HGCNumberingScheme : public CaloNumberingScheme {

public:
  HGCNumberingScheme(std::vector<double> gpar);
  virtual ~HGCNumberingScheme();
  virtual uint32_t getUnitID(int subdet, G4ThreeVector point, int iz, int mod,
                             int layer);

private:
  HGCNumberingScheme();

  std::vector<double>    gpar;
};

#endif
