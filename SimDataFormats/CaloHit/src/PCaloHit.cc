#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include<iostream>

//PG namespace cms
//PG {

  std::ostream & operator<<(std::ostream& o,const PCaloHit& hit)
  {
    o << "0x"<<std::hex<< hit.id() << std::dec
      << ": Energy " << hit.energy() << " GeV "
      << " Tof " << hit.time() << " ns "
      << " Geant track #" << hit.geantTrackId();

    return o;
  }

//PG } //PG namespace cms
