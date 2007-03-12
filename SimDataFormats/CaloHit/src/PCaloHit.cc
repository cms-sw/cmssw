#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include<iostream>

PCaloHit::PCaloHit(float eEM, float eHad, float t, int i) : myEnergyEM(eEM), 
							    myEnergyHad(eHad), 
							    myTime(t), 
							    myItra(i) { }

PCaloHit::PCaloHit(unsigned int id, float eEM, float eHad, float t, int i) :
  myEnergyEM(eEM), myEnergyHad(eHad), myTime(t), myItra(i), detId(id) { }


std::ostream & operator<<(std::ostream& o,const PCaloHit& hit)  {
  o << "0x"<<std::hex<< hit.id() << std::dec
    << ": Energy (EM) " << hit.energyEM() << " GeV "
    << ": Energy (Had) " << hit.energyHad() << " GeV "
    << " Tof " << hit.time() << " ns "
    << " Geant track #" << hit.geantTrackId();

  return o;
}
