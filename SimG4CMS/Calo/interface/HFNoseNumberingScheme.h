#ifndef SimG4CMS_Calo_HFNoseNumberingScheme_h
#define SimG4CMS_Calo_HFNoseNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: HFNoseNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for HFNose
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

#include "G4ThreeVector.hh"

class HFNoseNumberingScheme {
public:
  HFNoseNumberingScheme(const HGCalDDDConstants& hgc);
  ~HFNoseNumberingScheme() {}

  /**
     @short assigns the det id to a hit
   */
  uint32_t getUnitID(int layer, int module, int cell, int iz, const G4ThreeVector& pos, double& wt);

private:
  HFNoseNumberingScheme() = delete;
  const HGCalDDDConstants& hgcons_;
  const HGCalGeometryMode::GeometryMode mode_;
};

#endif
