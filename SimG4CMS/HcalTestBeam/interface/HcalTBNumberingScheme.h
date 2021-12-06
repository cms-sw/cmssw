#ifndef HcalTestBeam_HcalTBNumberingScheme_H
#define HcalTestBeam_HcalTBNumberingScheme_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTBNumberingScheme
//
/**\class HcalTBNumberingScheme HcalTBNumberingScheme.h SimG4CMS/HcalTestBeam/interface/HcalTBNumberingScheme.h
  
 Description: Numbering scheme for hadron calorimeter in test beam
  
 Usage: Sets up tower ID's of all towers in the 2004 Hcal test beam setup
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu May 18 10:14:34 CEST 2006
//

// system include files
#include <vector>
#include <cstdint>

// user include files

class HcalTBNumberingScheme {
public:
  HcalTBNumberingScheme() {}
  virtual ~HcalTBNumberingScheme() {}

  static uint32_t getUnitID(const uint32_t id, const int mode);
  static std::vector<uint32_t> getUnitIDs(const int type, const int mode);
};

#endif
