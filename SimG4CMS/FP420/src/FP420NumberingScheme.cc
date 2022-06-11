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

FP420NumberingScheme::~FP420NumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FP420") << " Deleting FP420NumberingScheme";
#endif
}

int FP420NumberingScheme::detectorLevel(const G4Step* aStep) const {
  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = 0;
  if (touch)
    level = ((touch->GetHistoryDepth()) + 1);
  return level;
}

void FP420NumberingScheme::detectorLevel(const G4Step* aStep, int& level, int* copyno, G4String* name) const {
  //Get name and copy numbers
  if (level > 0) {
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    for (int ii = 0; ii < level; ii++) {
      int i = level - ii - 1;
      name[ii] = touch->GetVolume(i)->GetName();
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}

unsigned int FP420NumberingScheme::getUnitID(const G4Step* aStep) const {
  unsigned intindex = 0;
  int level = detectorLevel(aStep);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FP420") << "FP420NumberingScheme number of levels= " << level;
#endif
  //  unsigned int intIndex = 0;
  if (level > 0) {
    int* copyno = new int[level];
    G4String* name = new G4String[level];
    detectorLevel(aStep, level, copyno, name);

    //    int det   = static_cast<int>(FP420);;

    int det = 1;
    int stationgen = 0;
    int zside = 0;
    int station = 0;
    int plane = 0;
    for (int ich = 0; ich < level; ich++) {
      /*
      // old set up configuration with equidistant stations 
      if(name[ich] == "FP420Ex") {
      stationgen   = copyno[ich];
      } else if(name[ich] == "SISTATION") {
      station   = stationgen;
      } else if(name[ich] == "SIPLANE") {
      plane   = copyno[ich];
      } else if(name[ich] == "SIDETL") {
      zside   = 1;
      } else if(name[ich] == "SIDETR") {
      zside   = 2;
      }
      */
      // new and old set up configurations are possible:
      if (name[ich] == "FP420E") {
        det = copyno[ich];
      } else if (name[ich] == "HPS240E") {
        det = copyno[ich] + 2;
      } else if (name[ich] == "FP420Ex1" || name[ich] == "HPS240Ex1") {
        stationgen = 1;
        //    } else if(name[ich] == "FP420Ex2") {
        //	stationgen   = 2;
      } else if (name[ich] == "FP420Ex3" || name[ich] == "HPS240Ex3") {
        stationgen = 2;  // was =3
      } else if (name[ich] == "SISTATION" || name[ich] == "HPS240SISTATION") {
        station = stationgen;
      } else if (name[ich] == "SIPLANE" || name[ich] == "HPS240SIPLANE") {
        plane = copyno[ich];
        // SIDETL (or R) can be ether X or Y type in next schemes of readout
        //        !!! (=...) zside
        //
        //      1                  2     <---copyno
        //   Front(=2) Empty(=4) Back(=6)     <--SIDETR OR SENSOR2
        //      1         2              <---copyno
        //   Front(=1) Back(=3) Empty(=5)     <--SIDETL OR SENSOR1
        //
      } else if (name[ich] == "SENSOR2" || name[ich] == "HPS240SENSOR2") {
        //    } else if(name[ich] == "SIDETR") {
        //	zside   = 4 * copyno[ich] - 2 ;//= 2   6 (copyno=1,2)
        zside = 3 * copyno[ich];  //= 3   6 (copyno=1,2)
      } else if (name[ich] == "SENSOR1" || name[ich] == "HPS240SENSOR1") {
        //     } else if(name[ich] == "SIDETL") {
        //	zside   = 2 * copyno[ich] - 1 ;//= 1   3 (copyno=1,2)
        zside = copyno[ich];  //= 1   2 (copyno=1,2)
      }
      //
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("FP420") << "FP420NumberingScheme  "
                                << "ich=" << ich << "copyno" << copyno[ich] << "name=" << name[ich];
#endif
    }
    // det = 1 for +FP420 ,  = 2 for -FP420  / (det-1) = 0,1
    // det = 3 for +HPS240 , = 4 for -HPS240 / (det-1) = 2,3
    // 0 is as default for every below:
    // Z index
    // station number 1 - 8   (in reality just 2 ones)
    // superplane(superlayer) number  1 - 16 (in reality just 5 ones)

    // intindex = myPacker.packEcalIndex (det, zside, station, plane);// see examples
    // intindex = myPacker.packCastorIndex (det, zside, station, plane);// see examples
    intindex = packFP420Index(det, zside, station, plane);
    //
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FP420") << "FP420NumberingScheme det=" << det << " zside=" << zside << " station=" << station
                              << " plane=" << plane;
    for (int ich = 0; ich < level; ich++) {
      edm::LogVerbatim("FP420") << " name = " << name[ich] << " copy = " << copyno[ich];
      edm::LogVerbatim("FP420") << " packed index = intindex" << intindex;
    }
#endif

    delete[] copyno;
    delete[] name;
  }

  return intindex;
}

unsigned FP420NumberingScheme::packFP420Index(int det, int zside, int station, int plane) {
  //
  unsigned int idx = ((det - 1) & 3) << 19;  //bit 19-20 (det-1):0- 3 = 4-->2**2 = 4  -> 4-1  ->((det-1)&3)  2 bit: 0-1
  // unsigned int idx = ((det-1)&1)<<20;//bit 20-20 (det-1):0- 1 = 2-->2**1 = 2  -> 2-1  ->((det-1)&1)  1 bit: 0
  idx += (zside & 7) << 7;  //bits  7- 9   zside:0- 7 = 8-->2**3 = 8  -> 8-1  ->  (zside&7)  3 bits:0-2
  //  idx += (zside&3)<<7;             //bits  7- 8   zside:0- 3 = 4-->2**2 = 4  -> 4-1  ->  (zside&3)  2 bits:0-1
  idx += (station & 7) << 4;  //bits  4- 6 station:0- 7 = 8-->2**3 = 8  -> 8-1  ->(station&7)  3 bits:0-2
  idx += (plane & 15);        //bits  0- 3   plane:0-15 =16-->2**4 =16  ->16-1  -> (plane&15)  4 bits:0-3

  //

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FP420") << "FP420 packing: det " << det << " zside  " << zside << " station " << station
                            << " plane " << plane << " idx " << idx;
#endif
  //  int newdet, newzside, newstation,newplane;
  //  unpackFP420Index(idx, newdet, newzside, newstation,newplane);

  //

  return idx;
}

void FP420NumberingScheme::unpackFP420Index(const unsigned int& idx, int& det, int& zside, int& station, int& plane) {
  det = (idx >> 19) & 3;
  //det  = (idx>>20)&1;
  det += 1;
  zside = (idx >> 7) & 7;
  //  zside   = (idx>>7)&3;
  station = (idx >> 4) & 7;
  plane = idx & 15;
  //

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FP420") << " FP420unpacking: idx=" << idx << " zside  " << zside << " station " << station
                            << " plane " << plane;
#endif
  //
}
