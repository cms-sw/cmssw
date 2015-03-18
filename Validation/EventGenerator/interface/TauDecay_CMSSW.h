// -*- C++ -*-
//
// Package:    TauNtuple
// Class:      TauDecay_CMSSW
// 
/**\class TauDecay TauDecay_CMSSW.cc TauDataFormat/TauNtuple/src/TauDecay_CMSSW.cc

 Description: Bridge class for TauDecay when using HepMC

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ian Nugent  
//         Created:  Fri Nov 18 13:49:02 CET 2011
//
//
#ifndef TauDecay_CMSSW_h
#define TauDecay_CMSSW_h

#include "Validation/EventGenerator/interface/TauDecay.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include <SimDataFormats/GeneratorProducts/interface/HepMCProduct.h>

//
// class declaration
//
class TauDecay_CMSSW : public TauDecay {
public:
  TauDecay_CMSSW();
  ~TauDecay_CMSSW();

  //Function to analyze the tau
  bool AnalyzeTau(HepMC::GenParticle *Tau,unsigned int &JAK_ID,unsigned int &TauBitMask,bool dores=true, bool dopi0=true);
  // Functions to get results
  std::vector<HepMC::GenParticle*> Get_TauDecayProducts(){return TauDecayProducts;}
  std::vector<unsigned int> Get_MotherIdx(){return MotherIdx;}

private:
  // recursive function to loop through tau decay products
  void Analyze(HepMC::GenParticle *Particle,unsigned int midx,bool dores, bool dopi0);
  void AddPi0Info(HepMC::GenParticle *Particle,unsigned int midx);
  //varibles
  std::vector<HepMC::GenParticle*> TauDecayProducts;
  std::vector<unsigned int> MotherIdx;
  unsigned int JAK_ID, TauBitMask;

};
#endif
