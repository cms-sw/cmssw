
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
using namespace edm;

SubEvent GenHIEvent::getSubEvent(int sub_id){
   SubEvent evt = evts_[sub_id]; 
   return evt;
}

std::vector<HepMC::GenParticle*> GenHIEvent::getParticles(HepMC::GenEvent evt, int sub_id){
   std::vector<HepMC::GenParticle*> particles = getSubEvent(sub_id).getParticles(evt);
   return particles;
}

HepMC::GenVertex* GenHIEvent::getVertex(HepMC::GenEvent evt, int sub_id){
   HepMC::GenVertex* vertex = getSubEvent(sub_id).getVertex(evt);
   return vertex;
}



