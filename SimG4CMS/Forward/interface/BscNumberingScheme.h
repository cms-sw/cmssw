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

class BscNumberingScheme {
public:
  BscNumberingScheme();
  ~BscNumberingScheme() = default;

  unsigned int getUnitID(const G4Step* aStep) const;

  static unsigned int packBscIndex(int det, int zside, int station);
  static void unpackBscIndex(const unsigned int& idx);
};

#endif
