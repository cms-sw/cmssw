#include <cstdint>
#ifndef SimDataFormats_HcalTestBeamNumbering_h
#define SimDataFormats_HcalTestBeamNumbering_h
///////////////////////////////////////////////////////////////////////////////
// File: HcalTestBeamNumbering.h
// Description: Numbering scheme for high granularity calorimeter (SIM step)
///////////////////////////////////////////////////////////////////////////////

class HcalTestBeamNumbering {
public:
  static const int kHcalBeamXValueOffset = 0;
  static const int kHcalBeamXValueMask = 0x1FF;
  static const int kHcalBeamXSignOffset = 9;
  static const int kHcalBeamXSignMask = 0x1;
  static const int kHcalBeamYValueOffset = 10;
  static const int kHcalBeamYValueMask = 0x1FF;
  static const int kHcalBeamYSignOffset = 19;
  static const int kHcalBeamYSignMask = 0x1;
  static const int kHcalBeamLayerOffset = 21;
  static const int kHcalBeamLayerMask = 0x7F;
  static const int kHcalBeamTypeOffset = 28;
  static const int kHcalBeamTypeMask = 0xF;

  enum HcalTestBeamDetector { HcalTBEmpty = 0, HcalTBScintillator = 1, HcalTBWireChamber = 2 };

  HcalTestBeamNumbering() {}
  static uint32_t packIndex(int det, int lay, int x, int y);
  static void unpackIndex(const uint32_t& idx, int& det, int& lay, int& x, int& y);
};

#endif
