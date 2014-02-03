#ifndef SimG4CMS_HGCNumberingScheme_h
#define SimG4CMS_HGCNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for HGC
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "G4Step.hh"
#include <boost/cstdint.hpp>
#include <vector>

class HGCNumberingScheme : public CaloNumberingScheme {

public:
  HGCNumberingScheme(std::vector<double> gpar);
  virtual ~HGCNumberingScheme();
  virtual uint32_t getUnitID(ForwardSubdetector &subdet, int &layer, int &module, int &iz, G4ThreeVector &pos, float &dz, float &bl1, float &tl1, float &h1);

private:
  HGCNumberingScheme();

  std::vector<double>    gpar;
};

#endif
