#ifndef SimDataFormats_SubEvent_h
#define SimDataFormats_SubEvent_h

#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenEvent.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"

namespace edm {

   class SubEvent {
   public:
      SubEvent(int id = -1): sub_id(id) {} 
      virtual ~SubEvent() {}
      std::vector<HepMC::GenParticle*>       getParticles(const HepMC::GenEvent& evt) const;
      HepMC::GenVertex*                   getVertex(const HepMC::GenEvent& evt) const;
      HepMC::GenParticle*                 getParticle(const HepMC::GenEvent& evt, int index) const;
      HepMC::GenParticle*                 getBoson(const HepMC::GenEvent& evt); 
      HepMC::GenParticle*                 getParton1(const HepMC::GenEvent& evt);
      HepMC::GenParticle*                 getParton2(const HepMC::GenEvent& evt);

   private:
      int                    sub_id;
   };
}

#endif
