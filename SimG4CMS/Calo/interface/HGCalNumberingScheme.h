#ifndef SimG4CMS_Calo_HGCalNumberingScheme_h
#define SimG4CMS_Calo_HGCalNumberingScheme_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCalNumberingScheme.h
// Description: Definition of sensitive unit numbering schema for HGC
///////////////////////////////////////////////////////////////////////////////

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

#include "G4ThreeVector.hh"

#include <string>
#include <vector>

class HGCalNumberingScheme {
public:
  HGCalNumberingScheme(const HGCalDDDConstants& hgc,
                       const DetId::Detector& det,
                       const std::string& name,
                       const std::string& fileName);
  HGCalNumberingScheme() = delete;
  ~HGCalNumberingScheme();

  /**
     @short assigns the det id to a hit
   */
  uint32_t getUnitID(int layer, int module, int cell, int iz, const G4ThreeVector& pos, double& wt);

private:
  void checkPosition(uint32_t index, const G4ThreeVector& pos, bool matchOnly, bool debug) const;

  const HGCalDDDConstants& hgcons_;
  const HGCalGeometryMode::GeometryMode mode_;
  const DetId::Detector det_;
  const std::string name_;
  int firstLayer_;
  std::vector<int> indices_;
  std::vector<int> dumpDets_;
  std::vector<int> dumpCassette_;
};

#endif
