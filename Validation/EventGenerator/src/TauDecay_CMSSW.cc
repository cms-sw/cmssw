#include "Validation/EventGenerator/interface/TauDecay_CMSSW.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"

#include <iomanip>
#include <cstdlib> 

TauDecay_CMSSW::TauDecay_CMSSW():
  TauDecay()
{

}

TauDecay_CMSSW::~TauDecay_CMSSW(){

} 
                            
bool TauDecay_CMSSW::AnalyzeTau(HepMC::GenParticle *Tau,unsigned int &JAK_ID,unsigned int &TauBitMask,bool dores, bool dopi0){
  Reset();
  MotherIdx.clear();
  TauDecayProducts.clear();
  if(abs(Tau->pdg_id())==PdtPdgMini::tau_minus){ // check that it is a tau
    unsigned int Tauidx=TauDecayProducts.size();
    HepMC::GenVertex::particle_iterator des;
    if( Tau->end_vertex()){
      for(des = Tau->end_vertex()->particles_begin(HepMC::children);
	  des!= Tau->end_vertex()->particles_end(HepMC::children);++des ) {
	Analyze((*des),Tauidx,dores,dopi0);
      }
      ClassifyDecayMode(JAK_ID,TauBitMask);
      return true;
    }
  }
  return false;
}



  
void TauDecay_CMSSW::Analyze(HepMC::GenParticle *Particle,unsigned int midx, bool dores, bool dopi0){
  unsigned int pdgid=abs(Particle->pdg_id());
  isTauResonanceCounter(pdgid);
  if(isTauFinalStateParticle(pdgid)){
    if(!isTauParticleCounter(pdgid)) std::cout << "TauDecay_CMSSW::Analyze WARNING: Unknow Final State Particle in Tau Decay... " << pdgid << std::endl;
    TauDecayProducts.push_back(Particle);
    MotherIdx.push_back(midx);
    return;
  }
  HepMC::GenVertex::particle_iterator des;
  if(Particle->end_vertex()){
    for(des = Particle->end_vertex()->particles_begin(HepMC::children); des!= Particle->end_vertex()->particles_end(HepMC::children) && Particle->end_vertex()-> particles_out_size()>0;++des ) {
      Analyze((*des),midx,dores,dopi0);
    }
  }
  else {
    std::cout << "Unstable particle that is undecayed in Tau decay tree. PDG ID: " << pdgid << std::endl;
  }
}



void TauDecay_CMSSW::AddPi0Info(HepMC::GenParticle *Particle,unsigned int midx){
  if(Particle->status()==1){
    TauDecayProducts.push_back(Particle);
    MotherIdx.push_back(midx);
    return;
  }
  HepMC::GenVertex::particle_iterator des;
  for(des = Particle->end_vertex()->particles_begin(HepMC::children);
      des!= Particle->end_vertex()->particles_end(HepMC::children);++des ) {
    AddPi0Info((*des),midx);
  }
}
