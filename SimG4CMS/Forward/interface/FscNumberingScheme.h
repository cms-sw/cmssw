///////////////////////////////////////////////////////////////////////////////
// File: FscNumberingScheme.h
// Date: 03.2026
// Description: Numbering scheme for Fsc
///////////////////////////////////////////////////////////////////////////////
#ifndef FscNumberingScheme_h
#define FscNumberingScheme_h

#include "G4Step.hh"
#include <cstdint>

namespace FscNumberingScheme {
  unsigned int getUnitID(const G4Step* aStep);

  unsigned int packFscIndex(int zside, int station, int phi);
  void unpackFscIndex(const unsigned int& idx, int& zside, int& stn, int& phi);
};  // namespace FscNumberingScheme

#endif
