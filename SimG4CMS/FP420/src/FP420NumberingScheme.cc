///////////////////////////////////////////////////////////////////////////////
// File: FP420NumberingScheme.cc
// Date: 02.2006
// Description: Numbering scheme for FP420
// Modifications: 08.2008  mside and fside added
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "globals.hh"
#include "G4Step.hh"
#include <iostream>

//#define EDM_ML_DEBUG

FP420NumberingScheme::FP420NumberingScheme() {
  //  sn0=3, pn0=6, rn0=7;
}

unsigned int FP420NumberingScheme::getUnitID(const G4Step* aStep) const {
  unsigned intindex = 0;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = (nullptr != touch) ? touch->GetHistoryDepth() + 1 : 0;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FP420") << "FP420NumberingScheme number of levels= " << level;
#endif
  if (level > 0) {
    int det = 1;
    int stationgen = 0;
    int zside = 0;
    int station = 0;
    int plane = 0;
    for (int ich = 0; ich < level; ich++) {
      int i = level - ich - 1;
      const G4String& name = touch->GetVolume(i)->GetName();
      int copyno = touch->GetReplicaNumber(i);

      // new and old set up configurations are possible:
      if (name == "FP420E") {
        det = copyno;
      } else if (name == "HPS240E") {
        det = copyno + 2;
      } else if (name == "FP420Ex1" || name == "HPS240Ex1") {
        stationgen = 1;
      } else if (name == "FP420Ex3" || name == "HPS240Ex3") {
        stationgen = 2;  // was =3
      } else if (name == "SISTATION" || name == "HPS240SISTATION") {
        station = stationgen;
      } else if (name == "SIPLANE" || name == "HPS240SIPLANE") {
        plane = copyno;
        // SIDETL (or R) can be ether X or Y type in next schemes of readout
        //        !!! (=...) zside
        //
        //      1                  2     <---copyno
        //   Front(=2) Empty(=4) Back(=6)     <--SIDETR OR SENSOR2
        //      1         2              <---copyno
        //   Front(=1) Back(=3) Empty(=5)     <--SIDETL OR SENSOR1
        //
      } else if (name == "SENSOR2" || name == "HPS240SENSOR2") {
        zside = 3 * copyno;  //= 3   6 (copyno=1,2)
      } else if (name == "SENSOR1" || name == "HPS240SENSOR1") {
        zside = copyno;  //= 1   2 (copyno=1,2)
      }
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("FP420") << "FP420NumberingScheme  "
                                << "ich=" << ich << " copyno=" << copyno << " name=" << name;
#endif
    }
    // det = 1 for +FP420 ,  = 2 for -FP420  / (det-1) = 0,1
    // det = 3 for +HPS240 , = 4 for -HPS240 / (det-1) = 2,3
    // 0 is as default for every below:
    // Z index
    // station number 1 - 8   (in reality just 2 ones)
    // superplane(superlayer) number  1 - 16 (in reality just 5 ones)

    intindex = packFP420Index(det, zside, station, plane);
    //
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FP420") << "FP420NumberingScheme det=" << det << " zside=" << zside << " station=" << station
                              << " plane=" << plane;
#endif
  }
  return intindex;
}

unsigned FP420NumberingScheme::packFP420Index(int det, int zside, int station, int plane) {
  unsigned int idx = ((det - 1) & 3) << 19;  //bit 19-20 (det-1):0- 3 = 4-->2**2 = 4  -> 4-1  ->((det-1)&3)  2 bit: 0-1
  idx += (zside & 7) << 7;                   //bits  7- 9   zside:0- 7 = 8-->2**3 = 8  -> 8-1  ->  (zside&7)  3 bits:0-2
  idx += (station & 7) << 4;                 //bits  4- 6 station:0- 7 = 8-->2**3 = 8  -> 8-1  ->(station&7)  3 bits:0-2
  idx += (plane & 15);                       //bits  0- 3   plane:0-15 =16-->2**4 =16  ->16-1  -> (plane&15)  4 bits:0-3

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FP420") << "FP420 packing: det " << det << " zside  " << zside << " station " << station
                            << " plane " << plane << " idx " << idx;
#endif

  return idx;
}

void FP420NumberingScheme::unpackFP420Index(const unsigned int& idx, int& det, int& zside, int& station, int& plane) {
  det = (idx >> 19) & 3;
  det += 1;
  zside = (idx >> 7) & 7;
  station = (idx >> 4) & 7;
  plane = idx & 15;
  //

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FP420") << " FP420unpacking: idx=" << idx << " zside  " << zside << " station " << station
                            << " plane " << plane;
#endif
  //
}
