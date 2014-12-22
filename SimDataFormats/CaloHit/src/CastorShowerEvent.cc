#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"
#include <iostream>

CastorShowerEvent::CastorShowerEvent() {
   // Clear();
   // std::cout << "\n    *** CastorShowerEvent object created ***    " << std::endl;
}

CastorShowerEvent::~CastorShowerEvent() {}

    
void CastorShowerEvent::Clear() {
   nhit = 0;
   detID.clear();
   hitPosition.clear();
   nphotons.clear();
   time.clear();
   primaryEnergy = 0.;
   primEta = 0.;
   primPhi = 0.;
   primX = 0.;
   primY = 0.;
   primZ = 0.;
}
