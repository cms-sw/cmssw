
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
using namespace edm;

SubEvent GenHIEvent::getSubEvent(int sub_id) const {
   SubEvent evt = evts_[sub_id]; 
   return evt;
}

std::vector<HepMC::GenParticle*> GenHIEvent::getParticles(const HepMC::GenEvent& evt, int sub_id) const {
   std::vector<HepMC::GenParticle*> particles = getSubEvent(sub_id).getParticles(evt);
   return particles;
}

const HepMC::GenVertex* GenHIEvent::getVertex(const HepMC::GenEvent& evt, int sub_id) const {
   const HepMC::GenVertex* vertex = getSubEvent(sub_id).getVertex(evt);
   return vertex;
}



