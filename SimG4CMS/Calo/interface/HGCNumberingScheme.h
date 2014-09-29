#ifndef SimG4CMS_HGCNumberingScheme_h
#define SimG4CMS_HGCNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for HGC
///////////////////////////////////////////////////////////////////////////////

#include "Geometry/CaloGeometry/interface/CaloNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "G4Step.hh"
#include <boost/cstdint.hpp>
#include <vector>

class HGCNumberingScheme : public CaloNumberingScheme {

public:

  enum HGCNumberingParameters { HGCCellSize };

  HGCNumberingScheme(const DDCompactView & cpv, std::string& name, bool check,
		     int verbose);

  virtual ~HGCNumberingScheme();

  /**
     @short assigns the det id to a hit
   */
  virtual uint32_t getUnitID(ForwardSubdetector subdet, int layer, int module, int iz, const G4ThreeVector &pos);

  /**
     @short maps a hit position to a sequential cell in a trapezoid surface defined by h,b,t
   */
  int assignCell(float x, float y, int layer);

  /**
     @short inverts the cell number in a trapezoid surface to local coordinates
   */
  std::pair<float,float> getLocalCoords(int cell, int layer);

  /**
     @short getter
   */
  const HGCalDDDConstants *getDDDConstants() { return hgcons; }

private:
  
  HGCNumberingScheme();

  bool                   check_;
  int                    verbosity;
  HGCalDDDConstants     *hgcons;
};

#endif
