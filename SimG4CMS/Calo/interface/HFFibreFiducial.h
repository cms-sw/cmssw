#ifndef SimG4CMS_HFFibreFiducial_h
#define SimG4CMS_HFFibreFiducial_h
///////////////////////////////////////////////////////////////////////////////
// File: HFFibreFiducial.h
// Description: Describes the fiducial limits of the fibres in HF
///////////////////////////////////////////////////////////////////////////////

#include "G4ThreeVector.hh"

class HFFibreFiducial {

public:
  HFFibreFiducial() {}
  virtual ~HFFibreFiducial() {}
  static int PMTNumber(G4ThreeVector pe_effect); // M.K. HF acceptance
};

#endif
