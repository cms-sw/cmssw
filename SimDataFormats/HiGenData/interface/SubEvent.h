#ifndef SimDataFormats_SubEvent_h
#define SimDataFormats_SubEvent_h

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenEvent.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"

namespace edm {

   class SubEvent {
   public:
      SubEvent(int id = -1): sub_id(id) {} 
      virtual ~SubEvent() {}
      std::vector<HepMC::GenParticle*>&       getParticles(HepMC::GenEvent& evt);
      HepMC::GenVertex*                   getVertex(HepMC::GenEvent& evt);
      HepMC::GenParticle*                 getParticle(HepMC::GenEvent& evt, int index);
      HepMC::GenParticle*                 getBoson(); 
      HepMC::GenParticle*                 getParton1();
      HepMC::GenParticle*                 getParton2();

   private:
      int                    sub_id;
   };
}

#endif
