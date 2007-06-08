///////////////////////////////////////////////////////////////////////////////
// File: FP420NumberingScheme.cc
// Date: 02.2006
// Description: Numbering scheme for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"
//
#include "CLHEP/Units/SystemOfUnits.h"
#include "globals.hh"
#include <iostream>

//#define debug

//UserVerbosity FP420NumberingScheme::cout("FP420NumberingScheme","silent","FP420NumberingScheme");

FP420NumberingScheme::FP420NumberingScheme() {
//  cout.infoOut << " Creating FP420NumberingScheme" << endl;
}

FP420NumberingScheme::~FP420NumberingScheme() {
//  cout.infoOut << " Deleting FP420NumberingScheme" << endl;
}

                                                                                
int FP420NumberingScheme::detectorLevel(const G4Step* aStep) const {
                                                                                
  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = 0;
  if (touch) level = ((touch->GetHistoryDepth())+1);
  return level;
}
                                                                                
void FP420NumberingScheme::detectorLevel(const G4Step* aStep, int& level,
                                        int* copyno, G4String* name) const {
                                                                                
  //Get name and copy numbers
  if (level > 0) {
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      name[ii]   = touch->GetVolume(i)->GetName();
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}
                                                                                



unsigned int FP420NumberingScheme::getUnitID(const G4Step* aStep) const {

  unsigned intindex=0;
  int level = detectorLevel(aStep);

#ifdef debug
//  cout.testOut << "FP420NumberingScheme number of levels= " << level << endl;
#endif

 //  unsigned int intIndex = 0;
  if (level > 0) {
    int*      copyno = new int[level];
    G4String* name   = new G4String[level];
    detectorLevel(aStep, level, copyno, name);

  //    int det   = static_cast<int>(FP420);;

    int stationgen  = 0;
    int zside   = 0;
    int station  = 0;
    int plane = 0;
    for (int ich=0; ich  <  level; ich++) {
      // pipe steel    
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
#ifdef debug
//      cout.testOut << "FP420NumberingScheme  " << "ich=" << ich  << "copyno" 
//		   << copyno[ich] << "name="  << name[ich] << endl;
#endif
     }
    // use for FP420 number 1 
    // 0 is as defauld for every below:
    // Z index X = 1; Y = 2 
    // station number 1 - 5
    // plane number  1 - 10

    int det = 1; 
   // intindex = myPacker.packEcalIndex (det, zside, station, plane);
   // intindex = myPacker.packCastorIndex (det, zside, station, plane);
    intindex = packFP420Index (det, zside, station, plane);


#ifdef debug
    /*
    cout.debugOut << "FP420NumberingScheme : det " << det << " zside " 
		  << zside << " station " << station << " plane " << plane
		  << " UnitID 0x" << hex << intindex << dec << endl;

    for (int ich = 0; ich < level; ich++)
      cout.debugOut <<" name = " << name[ich] <<" copy = " << copyno[ich] 
		    << endl;
    cout.testOut << " packed index = 0x" << hex << intindex << dec << endl;
*/
#endif

    delete[] copyno;
    delete[] name;
  }

  return intindex;
  
}

unsigned FP420NumberingScheme::packFP420Index(int det, int zside, int station,int plane){
  unsigned int idx = ((det-1)&1)<<20;     //bit 20
  idx += (zside&3)<<7;                  //bits 7-8    zside:0-2=3-->2**2 =4     2 bits:0-1
  idx += (station&7)<<4;                //bits 4-6   station:0-7=8-->2**3 =8     3 bits:0-2
  idx += (plane&15);                    //bits 0-3    plane:  0-15=16-->2**4 =16    4 bits:0-3
                                                                                
#ifdef debug
  /*
  cout.testOut << "FP420 packing: det " << det 
 << " zside  " << zside << " station " << station  << " plane " <<  plane << "-> 0x" << hex << idx << dec <<  endl;
  int newdet, newzside, newstation,newplane;
  unpackFP420Index(idx, newdet, newzside, newstation,newplane);
*/
#endif
                                                                                
  return idx;
}
void FP420NumberingScheme::unpackFP420Index(const unsigned int& idx, int& det,
                                        int& zside, int& station,
                                        int& plane) {
  det  = (idx>>20)&1;
  det += 1;
  zside   = (idx>>7)&3;
  station = (idx>>4)&7;
  plane   =  idx&15;
                                                                                
#ifdef debug
  /*
  cout.testOut  << " FP420 unpacking: 0x " << hex << idx << dec << " -> det " <<   det
          << " zside  " << zside << " station " << station  << " plane " <<  plane << endl;
*/
#endif
}
                                                                                
