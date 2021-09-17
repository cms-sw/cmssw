#include "SimG4CMS/Forward/interface/TotemT2ScintNumberingScheme.h"

uint32_t TotemT2ScintNumberingScheme::packID(const int& zside, const int& layer, const int& iphi) {
  uint32_t id = (((layer & kTotemT2LayerMask) << kTotemT2LayerOffset) | ((zside > 0) ? kTotemT2ZsideMask : 0) |
                 (iphi & kTotemT2PhiMask));
  return id;
}
