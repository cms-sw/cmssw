#include "Validation/EventGenerator/interface/TauDecay.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"
#include <iomanip>
#include <cstdlib> 
#include <iostream>
 
TauDecay::TauDecay(){
  Reset();
}

TauDecay::~TauDecay(){

}

void TauDecay::Reset(){
  n_pi=0;
  n_pi0=0;
  n_K=0;
  n_K0L=0;
  n_K0S=0;
  n_gamma=0;
  n_nu=0;
  n_e=0;
  n_mu=0;
  n_a1=0;
  n_a10=0;
  n_rho=0;
  n_rho0=0;
  n_eta=0;
  n_omega=0;
  n_Kstar0=0;
  n_Kstar=0;
  unknown=0;
}

bool TauDecay::isTauFinalStateParticle(int pdgid){
  int id=abs(pdgid);
  if(id==PdtPdgMini::e_minus)   return true;  // e+-
  if(id==PdtPdgMini::nu_e)      return true;  // nu_e
  if(id==PdtPdgMini::mu_minus)  return true;  // mu+-
  if(id==PdtPdgMini::nu_mu)     return true;  // nu_mu
  if(id==PdtPdgMini::nu_tau)    return true;  // nu_tau
  if(id==PdtPdgMini::gamma)     return true;  // gamma happends in generator
  if(id==PdtPdgMini::pi0)       return true;  // pi0
  if(id==PdtPdgMini::pi_plus)   return true;  // pi+-
  if(id==PdtPdgMini::K_L0)      return true;  // K0L
  if(id==PdtPdgMini::K_S0)      return true;  // KS
  if(id==PdtPdgMini::eta)       return true;
  if(id==PdtPdgMini::omega)     return true;

  if(id==PdtPdgMini::K_plus)    return true;  // K+-
  return false;
}

bool TauDecay::isTauParticleCounter(int pdgid){
  int id=abs(pdgid);
  //count particles
  if(id==PdtPdgMini::pi_plus) { n_pi++;       return true;}
  if(id==PdtPdgMini::pi0)     { n_pi0++;      return true;}
  if(id==PdtPdgMini::K_plus)  { n_K++;        return true;}
  if(id==PdtPdgMini::K_L0)    { n_K0L++;      return true;}
  if(id==PdtPdgMini::K_S0)    { n_K0S++;      return true;}
  if(id==PdtPdgMini::eta)     { n_eta++;     return true;}
  if(id==PdtPdgMini::omega)   { n_omega++;   return true;}
  if(id==PdtPdgMini::gamma)   { n_gamma++;    return true;}
  if(id==PdtPdgMini::nu_tau ||
     id==PdtPdgMini::nu_e   ||
     id==PdtPdgMini::nu_mu)    { n_nu++;      return true;}
  if(id==PdtPdgMini::e_minus)  { n_e++;       return true;}
  if(id==PdtPdgMini::mu_minus) { n_mu++;      return true;}
  if(abs(id)==PdtPdgMini::K0) std::cout << "TauDecay::isTauParticleCounter: ERROR unidentified Particle: " << id << std::endl;
  return false;
}

bool TauDecay::isTauResonanceCounter(int pdgid){
  int id=abs(pdgid);
  //count resonances
  if(id==PdtPdgMini::a_1_plus)   { n_a1++;      return true;}
  if(id==PdtPdgMini::a_10)       { n_a10++;     return true;}
  if(id==PdtPdgMini::rho_plus)   { n_rho++;     return true;}
  if(id==PdtPdgMini::rho0)       { n_rho0++;    return true;}
  if(id==PdtPdgMini::K_star0)    { n_Kstar0++;  return true;}
  if(id==PdtPdgMini::K_star_plus){ n_Kstar++;   return true;}
  if(id==PdtPdgMini::W_plus)     { return true;}
  unknown++;
  return false;
}

void TauDecay::ClassifyDecayMode(unsigned int &MODE_ID,unsigned int &TauBitMask){
  //Reset Bits
  MODE_ID=0;
  TauBitMask=0;
  // Classify according to MODE and TauDecayStructure
  if(n_pi+n_K+n_e+n_mu==1)TauBitMask=OneProng;
  if(n_pi+n_K==3)TauBitMask=ThreeProng;
  if(n_pi+n_K==5)TauBitMask=FiveProng;
  if(n_pi0==1)TauBitMask+=OnePi0;
  if(n_pi0==2)TauBitMask+=TwoPi0;
  if(n_pi0==3)TauBitMask+=ThreePi0;
  ClassifyDecayResonance(TauBitMask);

  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==0 && n_K==0 && n_K0S+n_K0L==2 && n_nu==1){
    MODE_ID=MODE_K0BK0PI; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==0 && n_K==0 && (n_K0L+n_K0S)==1 && n_nu==1){
    MODE_ID=MODE_K0PI; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==0 && n_pi0==1 && n_K==1 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_KPI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==1 && n_K==0 && n_K0L+n_K0S==1  && n_nu==1){
    MODE_ID=MODE_PIK0PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==0 && n_pi0==0 && n_K==1 && (n_K0L+n_K0S)==1 && n_nu==1){
    MODE_ID=MODE_KK0B; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==0 && n_pi0==1 && n_K==1 && n_K0L+n_K0S==1  && n_nu==1){
    MODE_ID=MODE_KK0BPI0; return;
  }
  if(n_e==1 && n_mu==0 && n_pi==0 && n_pi0==0 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==2){
    MODE_ID=MODE_ELECTRON; return;
  }
  if(n_e==0 && n_mu==1 && n_pi==0 && n_pi0==0 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==2){
    MODE_ID=MODE_MUON; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==0 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_PION; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==1 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_PIPI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==2 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_PI2PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==3 && n_pi0==0 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_3PI; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==0 && n_pi0==0 && n_K==1 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_KAON; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==3 && n_pi0==1 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_3PIPI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==3 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_PI3PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==3 && n_pi0==2 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_3PI2PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==5 && n_pi0==0 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_5PI; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==5 && n_pi0==1 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_5PIPI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==3 && n_pi0==3 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_3PI3PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==0 && n_K==2 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_KPIK; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==0 && n_pi0==2 && n_K==1 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_K2PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==2 && n_pi0==0 && n_K==1 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_KPIPI; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==1 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1 && n_gamma>=1 && n_rho==0){
    MODE_ID=MODE_PIPI0GAM; return ; // Obsolete should not be called
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==4 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_PI4PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==3 && n_pi0==0 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1 && n_eta==1){
    MODE_ID=MODE_3PIETA; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==2 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1 && n_eta==1){
    MODE_ID=MODE_PI2PI0ETA; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==2 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1 && n_omega==1){
    MODE_ID=MODE_PI2PI0OMEGA; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==3 && n_pi0==0 && n_K==0 && n_K0L==0 && n_K0S==0 && n_nu==1 && n_omega==1){
    MODE_ID=MODE_3PIOMEGA; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==0 && n_pi0==0 && n_K==1 && n_K0L==0 && n_K0S==0 && n_nu==1 && n_omega==1){
    MODE_ID=MODE_KOMEGA; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==0 && n_pi0==3 && n_K==1 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_K3PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==2 && n_pi0==1 && n_K==1 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_K2PIPI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==0 && n_pi0==0 && n_K==1 && n_K0L==0 && n_K0S==0 && n_nu==1 && n_eta==1){
    MODE_ID=MODE_KETA; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==2 && n_K==0 && (n_K0L+n_K0S)==0 && n_nu==1){
    MODE_ID=MODE_K0PI2PI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==3 && n_pi0==0 && n_K==0 && (n_K0L+n_K0S)==1 && n_nu==1){
    MODE_ID=MODE_K03PI; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==1 && n_K==0 && (n_K0L+n_K0S)==2 && n_nu==1){
    MODE_ID=MODE_2K0PIPI0; return;
  }
  if(n_e==0 && n_mu==0 && n_pi==1 && n_pi0==1 && n_K==2 && n_K0L==0 && n_K0S==0 && n_nu==1){
    MODE_ID=MODE_KPIKPI0; return;
  }
  if(n_pi==1 && n_pi0==1 && n_nu==1 && n_eta==1){ // eta modes
    MODE_ID=MODE_ETAPIPI0; return;
  }

  std::cout << "Tau Mode not found: n_e " <<  n_e << " n_mu " << n_mu << " n_pi " << n_pi << " n_pi0 " << n_pi0 << " n_K " << n_K << "  n_K0L " << n_K0L << "  n_K0S " << n_K0S << " n_nu  " << n_nu << " n_gamma " << n_gamma << std::endl;
  MODE_ID=MODE_UNKNOWN;
}

unsigned int TauDecay::nProng(unsigned int &TauBitMask){
  if(OneProng&TauBitMask)   return 1;
  if(ThreeProng&TauBitMask) return 3;
  if(FiveProng&TauBitMask)  return 5;
  return 7;
}
unsigned int TauDecay::nPi0(unsigned int &TauBitMask){
  if(OnePi0&TauBitMask)   return 1;
  if(TwoPi0&TauBitMask)   return 2;
  if(ThreePi0&TauBitMask) return 3;
  return 0;
}

bool TauDecay::hasResonance(unsigned int &TauBitMask, int pdgid){
  int p=abs(pdgid);
  if(p==PdtPdgMini::a_1_plus    && Res_a1_pm&TauBitMask)     return true;
  if(p==PdtPdgMini::a_10        && Res_a1_0&TauBitMask)      return true;
  if(p==PdtPdgMini::rho_plus    && Res_rho_pm&TauBitMask)    return true;
  if(p==PdtPdgMini::rho0        && Res_rho_0&TauBitMask)     return true;
  if(p==PdtPdgMini::eta         && Res_eta&TauBitMask)       return true;
  if(p==PdtPdgMini::omega       && Res_omega&TauBitMask)     return true;
  if(p==PdtPdgMini::K_star0     && Res_Kstar_pm&TauBitMask)  return true;
  if(p==PdtPdgMini::K_star_plus && Res_Kstar_0&TauBitMask)   return true;
  if(p==PdtPdgMini::K_S0        && KS0_to_pipi&TauBitMask)   return true;
  return false;
}


void TauDecay::ClassifyDecayResonance(unsigned int &TauBitMask){
  // Add Resonance info to TauBitMask
  if(n_a1>0)     TauBitMask+=Res_a1_pm;
  if(n_a10>0)    TauBitMask+=Res_a1_0;
  if(n_rho>0)    TauBitMask+=Res_rho_pm;
  if(n_rho0>0)   TauBitMask+=Res_rho_0;
  if(n_eta>0)    TauBitMask+=Res_eta;
  if(n_omega>0)  TauBitMask+=Res_omega;
  if(n_Kstar>0)  TauBitMask+=Res_Kstar_pm;
  if(n_Kstar0>0) TauBitMask+=Res_Kstar_0;
}

std::string TauDecay::DecayMode(unsigned int &MODE_ID){
  if(MODE_ID==MODE_ELECTRON)         return "#tau^{#pm} #rightarrow e^{#pm}#nu#nu";
  else if(MODE_ID==MODE_MUON)        return "#tau^{#pm} #rightarrow #mu^{#pm}#nu#nu";
  else if(MODE_ID==MODE_PION)        return "#tau^{#pm} #rightarrow #pi^{#pm}#nu";
  else if(MODE_ID==MODE_PIPI0)       return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{0}#nu";
  else if(MODE_ID==MODE_3PI)         return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{#pm}#pi^{#mp}#nu";
  else if(MODE_ID==MODE_PI2PI0)      return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{0}#pi^{0}#nu";
  else if(MODE_ID==MODE_KAON)        return "#tau^{#pm} #rightarrow K^{#pm}#nu";
  else if(MODE_ID==MODE_KPI0)        return "#tau^{#pm} #rightarrow K^{#pm}#pi^{0}#nu";
  else if(MODE_ID==MODE_K0PI)        return "#tau^{#pm} #rightarrow K^{0}#pi^{#pm}#nu";
  else if(MODE_ID==MODE_3PIPI0)      return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{#pm}#pi^{#mp}#pi^{0}#nu";
  else if(MODE_ID==MODE_PI3PI0)      return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{0}#pi^{0}#pi^{0}#nu";
  else if(MODE_ID==MODE_3PI2PI0)     return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{#pm}#pi^{#mp}#pi^{0}#pi^{0}#nu";
  else if(MODE_ID==MODE_5PI)         return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{#pm}#pi^{#pm}#pi^{#mp}#pi^{#mp}#nu";
  else if(MODE_ID==MODE_5PIPI0)      return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{#pm}#pi^{#pm}#pi^{#mp}#pi^{#mp}#pi^{0}#nu";
  else if(MODE_ID==MODE_3PI3PI0)     return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{#pm}#pi^{#mp}#pi^{0}#pi^{0}#pi^{0}#nu";
  else if(MODE_ID==MODE_KPIK)        return "#tau^{#pm} #rightarrow K^{#pm}#pi^{#pm}K^{#mp}#nu";
  else if(MODE_ID==MODE_K0BK0PI)     return "#tau^{#pm} #rightarrow #bar{K}^{0}K^{0}#pi^{#pm}#nu";
  else if(MODE_ID==MODE_KK0BPI0)     return "#tau^{#pm} #rightarrow #bar{K}^{0}K^{0}#pi^{#pm}#pi^{0}#nu";
  else if(MODE_ID==MODE_K2PI0)       return "#tau^{#pm} #rightarrow K^{#pm}#pi^{0}#pi^{0}#nu";
  else if(MODE_ID==MODE_KPIPI)       return "#tau^{#pm} #rightarrow K^{#pm}#pi^{#pm}#pi^{#mp}#nu";
  else if(MODE_ID==MODE_PIK0PI0)     return "#tau^{#pm} #rightarrow K^{0}#pi^{#pm}#pi^{0}#nu";
  else if(MODE_ID==MODE_ETAPIPI0)    return "#tau^{#pm} #rightarrow #pi^{#pm}#eta#pi^{0}#nu";
  else if(MODE_ID==MODE_PIPI0GAM)    return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{0}#nu#gamma (obsolete)";
  else if(MODE_ID==MODE_KK0B)        return "#tau^{#pm} #rightarrow K^{#pm}#bar{K}^{0}#nu";
  else if(MODE_ID==MODE_PI4PI0)      return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{0}#pi^{0}#pi^{0}#pi^{0}#nu";
  else if(MODE_ID==MODE_3PIETA)      return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{#pm}#pi^{#mp}#eta#nu";
  else if(MODE_ID==MODE_PI2PI0ETA)   return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{0}#pi^{0}#eta#nu";
  else if(MODE_ID==MODE_PI2PI0OMEGA) return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{0}#pi^{0}#omega#nu";
  else if(MODE_ID==MODE_3PIOMEGA)    return "#tau^{#pm} #rightarrow #pi^{#pm}#pi^{#pm}#pi^{#mp}#omega#nu";
  else if(MODE_ID==MODE_KOMEGA)      return "#tau^{#pm} #rightarrow K^{#pm}#omega#nu";
  else if(MODE_ID==MODE_K3PI0)       return "#tau^{#pm} #rightarrow K#pi^{0}#pi^{0}#pi^{0}#nu";
  else if(MODE_ID==MODE_K2PIPI0)     return "#tau^{#pm} #rightarrow K^{#pm}#pi^{#pm}#pi^{#mp}#pi^{0}#nu";
  else if(MODE_ID==MODE_KETA)        return "#tau^{#pm} #rightarrow K^{#pm}#eta#nu";
  else if(MODE_ID==MODE_K0PI2PI0)    return "#tau^{#pm} #rightarrow K^{0}#pi^{#pm}#pi^{0}#pi^{0}#nu";
  else if(MODE_ID==MODE_K03PI)       return "#tau^{#pm} #rightarrow K^{0}#pi^{#pm}#pi^{#pm}#pi^{#mp}#nu";
  else if(MODE_ID==MODE_2K0PIPI0)    return "#tau^{#pm} #rightarrow K^{0}#bar{K}^{0}#pi^{#pm}#pi^{-}#nu";
  else if(MODE_ID==MODE_KPIKPI0)     return "#tau^{#pm} #rightarrow K^{#pm}#pi^{#pm}K^{#mp}#pi^{0}#nu";
  return "UnKnown";
}
