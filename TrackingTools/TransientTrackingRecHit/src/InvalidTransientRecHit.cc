#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"


InvalidTransientRecHit::~InvalidTransientRecHit(){}

void InvalidTransientRecHit::throwError() const {
  throw cms::Exception("Invalid TrackingRecHit used");
}

GlobalPoint InvalidTransientRecHit::globalPosition() const { throwError(); return GlobalPoint();}
GlobalError InvalidTransientRecHit::globalPositionError() const { throwError(); return GlobalError();}

float InvalidTransientRecHit::errorGlobalR() const{ throwError(); return 0;}
float InvalidTransientRecHit::errorGlobalZ() const{ throwError(); return 0;}
float InvalidTransientRecHit::errorGlobalRPhi() const{ throwError(); return 0;}


AlgebraicVector InvalidTransientRecHit::parameters() const { 
  throwError();
  return AlgebraicVector();
}

AlgebraicSymMatrix InvalidTransientRecHit::parametersError() const { 
  throwError();
  return AlgebraicSymMatrix();
}
  
AlgebraicMatrix InvalidTransientRecHit::projectionMatrix() const { 
  throwError();
  return AlgebraicMatrix();
}

int InvalidTransientRecHit::dimension() const { throwError(); return 0;}

LocalPoint InvalidTransientRecHit::localPosition() const { 
  throwError();
  return LocalPoint();
}

LocalError InvalidTransientRecHit::localPositionError() const { 
  throwError();
  return LocalError();
}


std::vector<const TrackingRecHit*> InvalidTransientRecHit::recHits() const { 
  throwError();
  return std::vector<const TrackingRecHit*>();
}

std::vector<TrackingRecHit*> InvalidTransientRecHit::recHits() { 
  throwError();
  return std::vector<TrackingRecHit*>();
}

