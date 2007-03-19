#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include<iostream>

PCaloHit::PCaloHit(float eEM, float eHad, float t, int i) : myTime(t), 
							    myItra(i) { 
  myEnergy = eEM + eHad;
  myEMFraction = (myEnergy <= 0. ? 1. :eEM/myEnergy);
}

PCaloHit::PCaloHit(unsigned int id, float eEM, float eHad, float t, int i) :
  myTime(t), myItra(i), detId(id) {
  myEnergy = eEM + eHad;
  myEMFraction = (myEnergy <= 0. ? 1. :eEM/myEnergy);
}


std::ostream & operator<<(std::ostream& o,const PCaloHit& hit)  {
  o << "0x" <<std::hex << hit.id() << std::dec
    << ": Energy (EM) " << hit.energyEM() << " GeV "
    << ": Energy (Had) " << hit.energyHad() << " GeV "
    << " Tof " << hit.time() << " ns "
    << " Geant track #" << hit.geantTrackId();

  return o;
}
