///////////////////////////////////////////////////////////////////////////////
// File: EcalNumberingScheme.cc
// Description: Numbering scheme for electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/EcalNumberingScheme.h"

#include <iostream>

EcalNumberingScheme::EcalNumberingScheme(int iv) : CaloNumberingScheme(iv) {}

EcalNumberingScheme::~EcalNumberingScheme() {}

uint32_t EcalNumberingScheme::getUnitWithMaxEnergy(MapType& themap) {

  //look for max
  uint32_t unitWithMaxEnergy = 0;
  float    maxEnergy = 0.;
        
  MapType::iterator iter;
  for (iter = themap.begin(); iter != themap.end(); iter++) {
            
    if (maxEnergy < (*iter).second) {
      maxEnergy = (*iter).second;       
      unitWithMaxEnergy = (*iter).first;
    }                           
  }

  if (verbosity > 1)
    std::cout << "EcalNumberingScheme: *** max energy of " << maxEnergy 
	      << " MeV was found in Unit id 0x" << std::hex 
	      << unitWithMaxEnergy << std::dec << std::endl;
  return unitWithMaxEnergy;
}

