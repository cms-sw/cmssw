#ifndef SimG4CMSForwardBHMNumberingScheme_h
#define SimG4CMSForwardBHMNumberingScheme_h

#include "G4Step.hh"
#include <cstdint>

class BHMNumberingScheme {
public:
  BHMNumberingScheme();
  ~BHMNumberingScheme() = default;

  unsigned int getUnitID(const G4Step* aStep) const;

  static unsigned int packIndex(int subdet, int zside, int station);
  static void unpackIndex(const unsigned int& idx, int& subdet, int& zside, int& station);
};

#endif
