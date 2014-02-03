///////////////////////////////////////////////////////////////////////////////
// File: ZdcNumberingScheme.cc
// Date: 02.04
// Description: Numbering scheme for Zdc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#define debugLog

ZdcNumberingScheme::ZdcNumberingScheme(int iv){
  verbosity = iv;
  if (verbosity>0) 
    std::cout << "Creating ZDCNumberingScheme" << std::endl;
}

ZdcNumberingScheme::~ZdcNumberingScheme() {
  if (verbosity>0) 
    std::cout<< " Deleting ZdcNumberingScheme" <<  std::endl;
}


void ZdcNumberingScheme::setVerbosity(const int iv){ verbosity = iv; }

unsigned int ZdcNumberingScheme::getUnitID(const G4Step* aStep) const {

  uint32_t  index = 0;
  int level = detectorLevel(aStep);
 
  if (level > 0) {
    int*      copyno = new int[level];
    G4String* name   = new G4String[level];

    detectorLevel(aStep, level, copyno, name);

    int zside   = 0;
    int channel = 0;
    int fiber   = 0;
    int layer   = 0;
    HcalZDCDetId::Section section = HcalZDCDetId::Unknown;
    
    for (int ich=0; ich  <  level; ich++) {
      if (name[ich] == "ZDC") {
        if(copyno[ich] == 1)zside = 1;
        if(copyno[ich] == 2)zside = -1;
      } else if (name[ich] == "ZDC_EMLayer") {
        section = HcalZDCDetId::EM;
        layer = copyno[ich];
      } else if (name[ich] == "ZDC_EMFiber") {
        fiber = copyno[ich];
        if (fiber < 20)         channel = 1;
        else if (fiber < 39)    channel = 2;                         
        else if (fiber < 58)    channel = 3; 
        else if (fiber < 77)    channel = 4;
        else                    channel = 5;	
#ifdef debugLog
	std::cout<<"ZdcNumberingScheme::getUnitID section: "<<section<<", layer: "<<layer<<", channel: "<<channel<<", ich: "<<ich<<std::endl;
#endif
      } else if (name[ich] == "ZDC_LumLayer") {
        section = HcalZDCDetId::LUM;
        layer = copyno[ich];
        channel = layer;
      } else if (name[ich] == "ZDC_LumGas") {
        fiber = 1;
      } else if (name[ich] == "ZDC_HadLayer") {
        section = HcalZDCDetId::HAD;
        layer = copyno[ich];
      	if (layer < 6)          channel = 1;
      	else if (layer < 12)    channel = 2;
        else if (layer < 18)    channel = 3;
      	else                    channel = 4;
#ifdef debugLog
	std::cout<<"ZdcNumberingScheme::getUnitID section: "<<section<<", layer: "<<layer<<", channel: "<<channel<<", ich: "<<ich<<std::endl;
#endif
      } else if (name[ich] == "ZDC_HadFiber") {
        fiber = copyno[ich];
      } else if (name[ich] == "ZDC_FlowLayer") {
        section = HcalZDCDetId::RPD;
        layer = 1;
	channel=copyno[ich];
#ifdef debugLog
	std::cout<<"ZdcNumberingScheme::getUnitID section: "<<section<<", layer: "<<layer<<", channel: "<<channel<<", ich: "<<ich<<std::endl;
#endif
      } else if (name[ich] == "ZDC_FlowFiber") {
        fiber = 1;
      }
    }
 
#ifdef debugLog
    unsigned intindex=0;
    // intindex = myPacker.packZdcIndex (section, layer, fiber, channel, zside);
    intindex = packZdcIndex (section, layer, fiber, channel, zside);
    std::cout<<"ZdcNumberingScheme::getUnitID just packed section: "<<section<<", layer: "<<layer<<", fiber: "<<fiber<<", channel: "<<channel<<", zside: "<<zside<<", index: "<<intindex<<std::endl;
#endif

    bool true_for_positive_eta = true;
    //if(zside == 1)true_for_positive_eta = true;
    if(zside == -1)true_for_positive_eta = false;

    HcalZDCDetId zdcId(section, true_for_positive_eta, channel);
    index = zdcId.rawId();

#ifdef debugLog
    std::cout<<"DetectorId: ";
    std::cout<<zdcId<<std::endl;

    
    std::cout<< "ZdcNumberingScheme:" 
             << "  getUnitID - # of levels = " 
             << level << std::endl;
    for (int ich = 0; ich < level; ich++)
      std::cout<< " ZdcNumberingScheme: " << ich  << ": copyno " << copyno[ich] 
               << " name="  << name[ich]
               << "  section " << section << " zside " << zside
               << " layer " << layer << " fiber " << fiber
               << " channel " << channel << "packedIndex ="
               << intindex << " detId raw: "<<index<<std::endl;
               
#endif
    
    delete[] copyno;
    delete[] name;
  }
  
  return index;
  
}

unsigned ZdcNumberingScheme::packZdcIndex(int section, int layer, int fiber,
					  int channel, int z){
  unsigned int idx = ((z-1)&1)<<23;       //bit 23
  idx += (channel&31)<<18;                //bits 18-22
  idx += (fiber&255)<<10;                 //bits 10-17
  idx += (layer&127)<<3;                  //bits 3-9
  idx += (section&7);                     //bits 0-2

#ifdef debugLog
  std::cout<< "ZDC packing: section " << section << " layer  " << layer << " fiber "
	   << fiber  << " channel " <<  channel << " zside "  << z << "idx: " <<idx <<  std::endl;
#endif

  return idx;
}


void ZdcNumberingScheme::unpackZdcIndex(const unsigned int& idx, int& section,
					int& layer, int& fiber,
					int& channel, int& z) {

  z       = 1 + ((idx>>23)&1);
  channel = (idx>>18)&31;
  fiber   = (idx>>10)&255;
  layer   = (idx>>3)&127;
  section = idx&7;
	  
#ifdef debugLog
  std::cout<< "ZDC unpacking: idx:"<< idx << " -> section " << section
	   << " layer " << layer << " fiber " << fiber << " channel " 
	   << channel << " zside " << z <<  std::endl;
	   
  int newidx = packZdcIndex(section, layer, fiber, channel, z);
   std::cout<< "ZDC packing result after unpacking and repacking: section " << section << " layer  " << layer << " fiber "
	   << fiber  << " channel " <<  channel << " zside "  << z << " new idx: " <<newidx <<  std::endl;
#endif 
}

int ZdcNumberingScheme::detectorLevel(const G4Step* aStep) const {
  
  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = 0;
  if (touch) level = ((touch->GetHistoryDepth())+1);
  return level;
}
  
void ZdcNumberingScheme::detectorLevel(const G4Step* aStep, int& level,
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
