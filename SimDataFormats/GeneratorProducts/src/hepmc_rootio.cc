#include <HepMC/GenParticle.h>
#include <HepMC/GenVertex.h>
#include <iostream>

//HACK
namespace HepMC {
  struct GenEvent {
    static void clear_particles_in(HepMC::GenVertex* iVertex) {
      iVertex->m_particles_in.clear();
    }
    static void add_to_particles_in(HepMC::GenVertex* iVertex, HepMC::GenParticle* iPart) {
      iVertex->m_particles_in.push_back(iPart);
    }
  };
}


namespace hepmc_rootio {                                                                                                                
  void add_to_particles_in(HepMC::GenVertex* iVertex, HepMC::GenParticle* iPart) {
    HepMC::GenEvent::add_to_particles_in(iVertex, iPart);
  }

  void clear_particles_in(HepMC::GenVertex* iVertex) {
    HepMC::GenEvent::clear_particles_in(iVertex);
  }
}
