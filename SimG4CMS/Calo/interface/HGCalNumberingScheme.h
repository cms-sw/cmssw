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

#include "G4Step.hh"

class HGCalNumberingScheme {

public:

  HGCalNumberingScheme(const HGCalDDDConstants& hgc, 
		       const DetId::Detector& det, const std::string& name);
  ~HGCalNumberingScheme();

  /**
     @short assigns the det id to a hit
   */
  uint32_t getUnitID(int layer, int module, int cell, int iz,
		     const G4ThreeVector &pos, double& wt);

private:
  
  HGCalNumberingScheme() = delete;
  const HGCalDDDConstants&               hgcons_;
  const HGCalGeometryMode::GeometryMode  mode_;
  DetId::Detector                        det_;
  std::string                            name_;
};

#endif
