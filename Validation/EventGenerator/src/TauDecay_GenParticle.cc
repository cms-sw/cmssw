#include "Validation/EventGenerator/interface/TauDecay_GenParticle.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"

#include <iomanip>
#include <cstdlib>

TauDecay_GenParticle::TauDecay_GenParticle():
  TauDecay()
{

}

TauDecay_GenParticle::~TauDecay_GenParticle(){

}
                            
bool TauDecay_GenParticle::AnalyzeTau(const reco::GenParticle *Tau,unsigned int &JAK_ID,unsigned int &TauBitMask,bool dores, bool dopi0){
  Reset();
  MotherIdx.clear();
  TauDecayProducts.clear();
  if(abs(Tau->pdgId())==PdtPdgMini::tau_minus){ // check that it is a tau
    unsigned int Tauidx=TauDecayProducts.size();
    TauDecayProducts.push_back(Tau);
    MotherIdx.push_back(Tauidx);
    for (unsigned int i=0; i< Tau->numberOfDaughters(); i++){
      const reco::Candidate *dau=Tau->daughter(i);
      Analyze(static_cast<const reco::GenParticle*>(dau),Tauidx,dores,dopi0);
    }
    ClassifyDecayMode(JAK_ID,TauBitMask);
    return true;
  }
  return false;
}



  
void TauDecay_GenParticle::Analyze(const reco::GenParticle *Particle,unsigned int midx,bool dores, bool dopi0){
  unsigned int pdgid=abs(Particle->pdgId());
  if(isTauFinalStateParticle(pdgid)){
    if(!isTauParticleCounter(pdgid)) std::cout << "TauDecay_GenParticle::Analyze WARNING: Unknow Final State Particle in Tau Decay... " << std::endl;
    TauDecayProducts.push_back(Particle);
    MotherIdx.push_back(midx);
    if(pdgid==PdtPdgMini::pi0 && dopi0){// store information on pi0 decay products even though a pi0 is a finsal state particle (for 3PiPi0 studies)
      midx=MotherIdx.size()-1;
      for (unsigned int i=0; i< Particle->numberOfDaughters(); i++){
	const reco::Candidate *dau=Particle->daughter(i);
	AddPi0Info(static_cast<const reco::GenParticle*>(dau),midx);
      }
    }
    return;
  }
  if(Particle->status()==1 || (isTauResonanceCounter(pdgid) && dores)){
    TauDecayProducts.push_back(Particle);
    MotherIdx.push_back(midx);
    midx=MotherIdx.size()-1;
  }
  for (unsigned int i=0; i< Particle->numberOfDaughters(); i++){
    const reco::Candidate *dau=Particle->daughter(i);
    Analyze(static_cast<const reco::GenParticle*>(dau),midx,dores,dopi0);
  }
}


void TauDecay_GenParticle::AddPi0Info(const reco::GenParticle *Particle,unsigned int midx){
  if(Particle->status()==1){
    TauDecayProducts.push_back(Particle);
    MotherIdx.push_back(midx);
    return;
  }
  for (unsigned int i=0; i< Particle->numberOfDaughters(); i++){
    const reco::Candidate *dau=Particle->daughter(i);
    AddPi0Info(static_cast<const reco::GenParticle*>(dau),midx);
  }
}

