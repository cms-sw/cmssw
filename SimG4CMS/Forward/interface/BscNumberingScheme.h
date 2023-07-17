///////////////////////////////////////////////////////////////////////////////
// File: BscNumberingScheme.h
// Date: 02.2006
// Description: Numbering scheme for Bsc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#ifndef BscNumberingScheme_h
#define BscNumberingScheme_h

#include "G4Step.hh"
#include <cstdint>

namespace BscNumberingScheme {
  unsigned int getUnitID(const G4Step* aStep);

  unsigned int packBscIndex(int det, int zside, int station);
  void unpackBscIndex(const unsigned int& idx);
};  // namespace BscNumberingScheme

#endif
