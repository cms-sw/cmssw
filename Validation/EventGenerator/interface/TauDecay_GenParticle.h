// -*- C++ -*-
//
// Package: Validation/EventGenerator
// Class: TauDecay_GenParticle
/*
Description: Bridge class for TauDecay when using GenParticles  
Implementation:
[Notes on implementation]
*/
//
// Original Author: Ian Nugent
//

#ifndef TauDecay_GenParticle_h
#define TauDecay_GenParticle_h

#include "Validation/EventGenerator/interface/TauDecay.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include <SimDataFormats/GeneratorProducts/interface/HepMCProduct.h>

// class declaration
class TauDecay_GenParticle : public TauDecay {
 public:
  TauDecay_GenParticle();
  ~TauDecay_GenParticle();

  //Function to analyze the tau
  bool AnalyzeTau(const reco::GenParticle *Tau,unsigned int &JAK_ID,unsigned int &TauBitMask,bool dores, bool dopi0);
  // Functions to get results
  std::vector<const reco::GenParticle* > Get_TauDecayProducts(){return TauDecayProducts;}
  std::vector<unsigned int> Get_MotherIdx(){return MotherIdx;}

 private:
  // recursive function to loop through tau decay products
  void Analyze(const reco::GenParticle *Particle,unsigned int midx,bool dores, bool dopi0);
  void AddPi0Info(const reco::GenParticle *Particle,unsigned int midx);
  //varibles
  std::vector<const reco::GenParticle*> TauDecayProducts;
  std::vector<unsigned int> MotherIdx;
  unsigned int JAK_ID, TauBitMask;

 };
#endif
