///////////////////////////////////////////////////////////////////////////////
// File: CastorNumberingScheme.cc
// Description: Numbering scheme for Castor
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "G4LogicalVolumeStore.hh"
#include <iostream>

//#define castornumschemedebug

CastorNumberingScheme::CastorNumberingScheme(): lvCASTFar(nullptr),lvCASTNear(nullptr),
                                                lvCAST(nullptr),lvCAES(nullptr),lvCEDS(nullptr),
						lvCAHS(nullptr),lvCHDS(nullptr),lvCAER(nullptr),
						lvCEDR(nullptr),lvCAHR(nullptr),lvCHDR(nullptr),
						lvC3EF(nullptr),lvC3HF(nullptr),lvC4EF(nullptr),
						lvC4HF(nullptr) {
  edm::LogInfo("ForwardSim") << "Creating CastorNumberingScheme";
  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<lvp>::const_iterator lvcite;
  for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
    if (strcmp(((*lvcite)->GetName()).c_str(),"CASTFar") == 0) lvCASTFar = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CASTNear") == 0) lvCASTNear = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CAST") == 0) lvCAST = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CAES") == 0) lvCAES = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CEDS") == 0) lvCEDS = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CAHS") == 0) lvCAHS = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CHDS") == 0) lvCHDS = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CAER") == 0) lvCAER = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CEDR") == 0) lvCEDR = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CAHR") == 0) lvCAHR = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"CHDR") == 0) lvCHDR = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"C3EF") == 0) lvC3EF = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"C3HF") == 0) lvC3HF = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"C4EF") == 0) lvC4EF = (*lvcite);
    if (strcmp(((*lvcite)->GetName()).c_str(),"C4HF") == 0) lvC4HF = (*lvcite);
  }
#ifdef castornumschemedebug
  LogDebug("ForwardSim") << "CastorNumberingScheme:: LogicalVolume pointers\n"
                         << lvCASTFar << " for CASTFar; " << lvCASTNear << " for CASTNear; "
			 << lvCAST << " for CAST; " << lvCAES << " for CAES; "
			 << lvCEDS << " for CEDS; " << lvCAHS << " for CAHS; "
			 << lvCHDS << " for CHDS; " << lvCAER << " for CAER; "
			 << lvCEDR << " for CEDR; " << lvCAHR << " for CAHR; "
			 << lvCHDR << " for CHDR; " << lvC3EF << " for C3EF; "
			 << lvC3HF << " for C3HF; " << lvC4EF << " for C4EF; "
	     << lvC4HF << " for C4HF.";

  LogDebug("ForwardSim") << "Call to init CastorNumberingScheme\n";
  for (int mod=0; mod<15; mod++)
    for (int sec=0; sec<17; sec++)
      {
	HcalCastorDetId castorId = HcalCastorDetId(false, sec, mod);
	LogDebug("ForwardSim") << "Mod: " << mod << "  Sec: " << sec << "  Id: " << castorId.rawId() << "\n";
      }

#endif
   
}

CastorNumberingScheme::~CastorNumberingScheme() {
}

uint32_t CastorNumberingScheme::getUnitID(const G4Step* aStep) const {

  uint32_t index=0;
  int      level, copyno[20];
  lvp      lvs[20];
  detectorLevel(aStep, level, copyno, lvs);

#ifdef castornumschemedebug
  LogDebug("ForwardSim") << "CastorNumberingScheme number of levels= " <<level;
#endif

  if (level > 0) {

    int zside   = 0;
    int sector  = 0;
    int module = 0;

    bool farSide = false;
    int castorGeoVersion = 0; //0 = original // 1 = separated-halves geometry

    //    HcalCastorDetId::Section section;
    for (int ich=0; ich < level; ich++) {
      if(lvs[ich] == lvCAST) {
        // Z index +Z = 1 ; -Z = 2
	assert (1 <= copyno[ich] && copyno[ich] <= 3);
        zside = copyno[ich] == 1 ? 1 : 2;
      } // copyno 2 = Far : 3 = Near
      else if(lvs[ich] == lvCASTFar || lvs[ich] == lvCASTNear) {
	castorGeoVersion = 1; //detected separated-halves geometry
	if(lvs[ich] == lvCASTFar)
	  farSide = true;
      }
      else if(lvs[ich] == lvCAES || lvs[ich] == lvCEDS ||
	      lvs[ich] == lvCAHS || lvs[ich] == lvCHDS) {
	// sector number for dead material 1 - 8
	int copyn = copyno[ich];
	if(castorGeoVersion == 1) {
          //for separated-half geometry the copy numbers do not start at "3 o'clock" and go from 1-8.
          //instead they start at "12 o'clock" for near side 1-4. and "6 o'clock" for far side 1-4 again
          if(farSide) {
            if (copyn<3)
              copyn += 6; //maps 1->7, 2->8
            else
              copyn -= 2; //maps 3->1 and 4->2
          }
          else { //nearSide
            copyn += 2; //maps 1->3, ...
          }
        } //endif separated-half geometry
	if (copyn<5)
	  sector = 5-copyn;
	else
	  sector = 13-copyn;
      }
      else if(lvs[ich] == lvCAER || lvs[ich] == lvCEDR) {
	// zmodule number 1-2 for EM section (2 copies)
	module = copyno[ich];
      }
      else if(lvs[ich] == lvCAHR || lvs[ich] == lvCHDR) {
	//zmodule number 3-14 for HAD section (12 copies)
	module = copyno[ich] + 2;
      }
      else if(lvs[ich] == lvC3EF || lvs[ich] == lvC3HF) {
	// sector number for sensitive material 1 - 16
	sector = sector*2;
      }
      else if(lvs[ich] == lvC4EF || lvs[ich] == lvC4HF) {
	// sector number for sensitive material 1 - 16
	sector = sector*2 - 1;
      }
    
#ifdef castornumschemedebug
      LogDebug("ForwardSim") << "CastorNumberingScheme  " << "ich = " << ich  
			     << "copyno = " << copyno[ich] << "name = " 
                             << lvs[ich]->GetName();
#endif
    } //end for loop over levels 

    // use for Castor number det = 9
    //
    // Z index +Z = 1 ; -Z = 2
    // sector number 1 - 16
    // zmodule number  1 - 18

    bool true_for_positive_eta = false;
    if(zside == 1) true_for_positive_eta = true;

    HcalCastorDetId castorId = HcalCastorDetId(true_for_positive_eta, sector, module);
    index = castorId.rawId();

#ifdef castornumschemedebug
    uint32_t intindex = 0;
    intindex = packIndex(zside, sector, module);
    LogDebug("ForwardSim") << "CastorNumberingScheme: " << " zside "
                           << zside << " module " << module << " sector " 
                           << sector << " UnitID 0x" << std::hex << intindex 
                           << std::dec << " index: " << index;
#endif
  }
  return index;

}

uint32_t CastorNumberingScheme::packIndex(int z, int sector, int module) {
  /*
    uint32_t idx=(section&31)<<28;     //bits 28-31   (21-27 are free for now)
    idx+=((z-1)&1)<<20;                //bits  20  (1...2)
    idx+=(sector&15)<<6;               //bits  6-9 (1...16)
    idx+=(module&63);                 //bits  0-5 (1...18)
    return idx;
  */

  uint32_t idx=((z-1)&1)<<8;       //bit 8
  idx+=(sector&15)<<4;             //bits  4-7 (1...16)
  idx+=(module&15);                 //bits  0-3 (1...14)
  return idx;

}

void CastorNumberingScheme::unpackIndex(const uint32_t& idx, int& z, int& sector, int& module) {
  /*
    section = (idx>>28)&31;
    z   = (idx>>20)&1;
    z  += 1;
    sector = (idx>>6)&15;
    module= (idx&63);
  */
  z   = (idx>>8)&1;
  z  += 1;
  sector = (idx>>4)&15;
  module = (idx&15);
}

void CastorNumberingScheme::detectorLevel(const G4Step* aStep, int& level,
					  int* copyno, lvp* lvs) const {

  //Get name and copy numbers
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  level = 0;
  if (touch) level = ((touch->GetHistoryDepth())+1);
  if (level > 0) {
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      lvs[ii]    = touch->GetVolume(i)->GetLogicalVolume();
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}
