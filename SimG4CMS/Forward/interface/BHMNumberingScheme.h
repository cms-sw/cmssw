#ifndef SimG4CMSForwardBHMNumberingScheme_h
#define SimG4CMSForwardBHMNumberingScheme_h

#include "G4Step.hh"
#include <vector>

namespace BHMNumberingScheme {
  unsigned int getUnitID(const G4Step* aStep);

  unsigned int packIndex(int subdet, int zside, int station);
  void unpackIndex(const unsigned int& idx, int& subdet, int& zside, int& station);
}  // namespace BHMNumberingScheme

#endif
