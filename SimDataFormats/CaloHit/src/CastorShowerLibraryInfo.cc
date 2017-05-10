#include "SimDataFormats/CaloHit/interface/CastorShowerLibraryInfo.h"
#include <iostream>

CastorShowerLibraryInfo::CastorShowerLibraryInfo() {
   // Clear();
   // std::cout << "\n    *** CastorShowerLibraryInfo object created ***    " << std::endl;
}

CastorShowerLibraryInfo::~CastorShowerLibraryInfo() {}

    
void CastorShowerLibraryInfo::ClearB() {
   Energy.ClearB();
   Eta.ClearB();
   Phi.ClearB();
}
