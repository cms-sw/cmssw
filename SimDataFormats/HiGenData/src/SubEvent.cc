#include <iostream> 
#include <iterator>
#include "SimDataFormats/HiGenData/interface/SubEvent.h"
using namespace edm;

HepMC::GenParticle* SubEvent::getParticle(const HepMC::GenEvent& evt, int index) const {

  //to be fixed
   
   std::vector<HepMC::GenParticle*> parts = getParticles(evt);
   HepMC::GenParticle* particle = parts[index];
   if(particle) return particle;
   else{
      std::cout<<"error loading particle, particle with index "<<index<<" doesn't exist!"<<std::endl;
      return 0;
   }
}

std::vector<HepMC::GenParticle*> SubEvent::getParticles(const HepMC::GenEvent& evt) const {
      std::vector<HepMC::GenParticle*> cands;

   HepMC::GenVertex* vertex = getVertex(evt);
   HepMC::GenVertex::particle_iterator p;
   HepMC::GenVertex::particle_iterator start = vertex->particles_begin( HepMC::relatives ); //
   HepMC::GenVertex::particle_iterator end = vertex->particles_end( HepMC::relatives ); //
   for ( p = start; p != end; ++p ) {
      cands.push_back(*p);
   }
	
      return cands;

}

HepMC::GenVertex*          SubEvent::getVertex(const HepMC::GenEvent& evt) const {
   
   HepMC::GenVertex* vertex;
   HepMC::GenEvent::vertex_const_iterator v;
   HepMC::GenEvent::vertex_const_iterator start = evt.vertices_begin();
   HepMC::GenEvent::vertex_const_iterator end = evt.vertices_end();
   for ( v = start; v != end; ++v ){
      HepMC::GenVertex* dummy = *v;
      if(dummy->id() == sub_id){
	 vertex = *v; 
	 break;
      }
   }
   if(!vertex){
      std::cout<<"Error - Vertex with id : "<<sub_id<<" could not be found!"<<std::endl;
   }
   return vertex;
}
   
HepMC::GenParticle*                SubEvent::getBoson(const HepMC::GenEvent& evt){
   HepMC::GenParticle* boson = getParticle(evt,3);
      if(boson) return boson;
      else{
	 std::cout<<"error loading boson, particle with index 1 doesn't exist!"<<std::endl;
	 return 0;
      }
}
   HepMC::GenParticle*                SubEvent::getParton1(const HepMC::GenEvent& evt){
      HepMC::GenParticle* parton = getParticle(evt,1);
      if(parton) return parton;
      else{
	 std::cout<<"error loading parton, particle with index 2 doesn't exist!"<<std::endl;
         return 0;
      }
}
   HepMC::GenParticle*                 SubEvent::getParton2(const HepMC::GenEvent& evt){
      HepMC::GenParticle* parton = getParticle(evt,2);
      if(parton) return parton;
      else{
	 std::cout<<"error loading parton, particle with index 3 doesn't exist!"<<std::endl;
         return 0;
      }
}
