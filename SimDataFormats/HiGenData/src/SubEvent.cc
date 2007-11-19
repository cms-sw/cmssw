 
#include <iterator>
#include "SimDataFormats/HiGenData/interface/SubEvent.h"
using namespace edm;

HepMC::GenParticle* SubEvent::getParticle(HepMC::GenEvent& evt, int index){

  //to be fixed
   /*
   std::vector<HepMC::GenParticle*>& parts = getParticles(evt);
   HepMC::GenParticle* particle = parts[index];
   return particle;
   */
   return 0;
}

std::vector<HepMC::GenParticle*>& SubEvent::getParticles(HepMC::GenEvent& evt){
      std::vector<HepMC::GenParticle*> cands;

      //to be fixed
	/*
   HepMC::GenVertex* vertex = getVertex(evt);
   HepMC::GenVertex::particle_iterator p;
   HepMC::GenVertex::particle_iterator start = vertex->particles_begin( HepMC::IteratorRange range=family ); //
   HepMC::GenVertex::particle_iterator end = vertex->particles_end( HepMC::IteratorRange range=family ); //
   for ( p = start; p != end; ++p ) {
      cands.push_back(*p);
   }
	*/
      return cands;

}

HepMC::GenVertex*                SubEvent::getVertex(HepMC::GenEvent& evt){

   HepMC::GenVertex* vertex;
   HepMC::GenEvent::vertex_const_iterator v;
   HepMC::GenEvent::vertex_const_iterator start = evt.vertices_begin();
   HepMC::GenEvent::vertex_const_iterator end = evt.vertices_end();
   for ( v = start; v != end; ++v ){
      HepMC::GenVertex* dummy = *v;
      if(dummy->id() == sub_id){
	 vertex = dummy; 
      }
      return vertex;
   }
}
   HepMC::GenParticle*                SubEvent::getBoson(){
   return 0;
}
   HepMC::GenParticle*                SubEvent::getParton1(){
   return 0;
}
   HepMC::GenParticle*                 SubEvent::getParton2(){
   return 0;
}
