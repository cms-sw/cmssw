#include "SimDataFormats/CaloHit/interface/CastorShowerLibraryInfo.h"
#include <iostream>

ClassImp(CastorShowerLibraryInfo)

CastorShowerLibraryInfo::CastorShowerLibraryInfo() {
   // Clear();
   // std::cout << "\n    *** CastorShowerLibraryInfo object created ***    " << std::endl;
}

CastorShowerLibraryInfo::~CastorShowerLibraryInfo() {}

    
void CastorShowerLibraryInfo::Clear() {
   NEv = 0;
   NEnBins = 0;
   NEvPerBin = 0;
   Energies.clear();
}
