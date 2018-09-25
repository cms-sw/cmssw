#ifndef SimG4CMS_HFFibreFiducial_h
#define SimG4CMS_HFFibreFiducial_h
///////////////////////////////////////////////////////////////////////////////
// File: HFFibreFiducial.h
// Description: Describes the fiducial limits of the fibres in HF
///////////////////////////////////////////////////////////////////////////////

#include "G4ThreeVector.hh"

namespace HFFibreFiducial {
  int PMTNumber(const G4ThreeVector& pe_effect); // M.K. HF acceptance
};

#endif
