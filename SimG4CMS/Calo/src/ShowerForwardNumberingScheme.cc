///////////////////////////////////////////////////////////////////////////////
// File: ShowerForwardNumberingScheme.cc
// Description: Numbering scheme for preshower detector
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/ShowerForwardNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#define debug

using namespace std;

ShowerForwardNumberingScheme::ShowerForwardNumberingScheme(int iv) : 
  EcalNumberingScheme(iv) {
  if (verbosity>0) 
    std::cout << "Creating ShowerForwardNumberingScheme" << std::endl;
}

ShowerForwardNumberingScheme::~ShowerForwardNumberingScheme() {
  if (verbosity>0) 
    std::cout << "Deleting ShowerForwardNumberingScheme" << std::endl;
}

uint32_t ShowerForwardNumberingScheme::getUnitID(const G4Step* aStep) const {

  int level = detectorLevel(aStep);
  uint32_t intIndex = 0;
  if (level > 0) {
    int*      copyno = new int[level];
    G4String* name   = new G4String[level];
    detectorLevel(aStep, level, copyno, name);


    // depth index - silicon layer 1-st or 2-nd
    int layer = 0;
    if(name[level-1] == "SFSX") {
      layer = 1;
    } else if (name[level-1] == "SFSY") {
      layer = 2;
    } else {
      std::cout << "ShowerForwardNumberingScheme: Wrong name of Presh. Si."
		   << " Strip : " << name[level-1] << std::endl;
    }
    // Z index +Z = 1 ; -Z = 2
    int zside   = copyno[level-6];
    zside=2*(1-zside)+1;
    // wafer number
    int wafer = copyno[level-4];
    // strip number inside wafer
    int strip = copyno[level-1];

    intIndex =  ESDetId(1,1,1,1,1).rawId(); //Fake for the moment
#ifdef debug
    if (verbosity>1) {
      std::cout << "ShowerForwardNumberingScheme : zside " 
		<< zside << " layer " << layer << " wafer " << wafer 
		<< " strip " << strip << " UnitID 0x" << std::hex << intIndex 
		<< std::dec << std::endl;
      for (int ich = 0; ich < level; ich++) {
	std::cout << "Name = " << name[ich] << " copy = " << copyno[ich]
		  << std::endl;
      }
    }
#endif
    delete[] copyno;
    delete[] name;
  }
  return intIndex;

}
