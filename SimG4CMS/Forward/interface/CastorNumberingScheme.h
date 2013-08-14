#ifndef Forward_CastorNumberingScheme_h
#define Forward_CastorNumberingScheme_h
// -*- C++ -*-
//
// Package:     Forward
// Class  :     CastorNumberingScheme
//
/**\class CastorNumberingScheme CastorNumberingScheme.h SimG4CMS/Forward/interface/CastorNumberingScheme.h
 
 Description: This class manages the UnitID that labels Castor sensitive
              volumes
 
 Usage: Used in CastorSD to get unique ID of sensitive detector element
 
*/
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: CastorNumberingScheme.h,v 1.5 2009/09/02 20:41:25 sunanda Exp $
//
 
// system include files

// user include files


#include "G4Step.hh"
#include "G4LogicalVolume.hh"
#include <boost/cstdint.hpp>

class CastorNumberingScheme {

public:

  CastorNumberingScheme();
  virtual ~CastorNumberingScheme();

  virtual uint32_t getUnitID(const G4Step* aStep) const;

  /** pack the Unit ID for  Castor <br>
   Bits  0- 5: zmodule index <br>
   Bits  6- 9: sector  index <br>
   Bits 10-19: unused        <br>
   Bit     20: +/- z side    <br>
   Bits 22-27: unused        <br>
   Bits 28-31: subdetector   <br>
   *  (+z=1,-z=2);  sector=1..16, zmodule=1..18;
   */
  //  static uint32_t packIndex(int det, int z, int sector, int zmodule);
  //  static void   unpackIndex(const uint32_t& idx, int& det, int& z, int& sector, int& zmodule);


  static uint32_t packIndex(int z, int sector, int zmodule);
  static void   unpackIndex(const uint32_t& idx, int& z, int& sector, int& zmodule);

private:

  typedef G4LogicalVolume* lvp;

  // Utilities to get detector levels during a step
  void detectorLevel(const G4Step*, int&, int*, lvp*) const;

  lvp lvCAST, lvCAES, lvCEDS, lvCAHS, lvCHDS, lvCAER, lvCEDR;
  lvp lvCAHR, lvCHDR, lvC3EF, lvC3HF, lvC4EF, lvC4HF;

};

#endif
