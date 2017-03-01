#ifndef SimG4CMS_HGCNumberingScheme_h
#define SimG4CMS_HGCNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for HGC
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "G4Step.hh"

class HGCNumberingScheme {

public:

  enum HGCNumberingParameters { HGCCellSize };

  HGCNumberingScheme(const HGCalDDDConstants& hgc, std::string& name );

  ~HGCNumberingScheme();

  /**
     @short assigns the det id to a hit
   */
  uint32_t getUnitID(ForwardSubdetector subdet, int layer, int module,
		     int cell, int iz, const G4ThreeVector &pos);

  /**
     @short maps a hit position to a sequential cell in a trapezoid surface defined by h,b,t
   */
  int assignCell(float x, float y, int layer);

  /**
     @short inverts the cell number in a trapezoid surface to local coordinates
   */
  std::pair<float,float> getLocalCoords(int cell, int layer);

private:
  
  HGCNumberingScheme();
  const HGCalDDDConstants& hgcons_;
};

#endif
