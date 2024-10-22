#ifndef Forward_TotemT2ScintNumberingScheme_h
#define Forward_TotemT2ScintNumberingScheme_h 1

#include <cstdint>

class TotemT2ScintNumberingScheme {
public:
  static constexpr uint32_t kTotemT2PhiMask = 0xF;
  static constexpr uint32_t kTotemT2LayerOffset = 4;
  static constexpr uint32_t kTotemT2LayerMask = 0x7;
  static constexpr uint32_t kTotemT2ZsideMask = 0x80;

  TotemT2ScintNumberingScheme() {}

  static uint32_t packID(const int& zside, const int& layer, const int& iphi);
  static int zside(const uint32_t& id) { return (id & kTotemT2ZsideMask) ? (1) : (-1); }
  static int layer(const uint32_t& id) { return ((id >> kTotemT2LayerOffset) & kTotemT2LayerMask); }
  static int iphi(const uint32_t& id) { return (id & kTotemT2PhiMask); }
};

#endif
