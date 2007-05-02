// -*- C++ -*-

#include "TopQuarkAnalysis/TopKinFitter/src/TKinFitter.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TSLToyGen.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintEp.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintMGaus.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitConstraintM.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleCart.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleECart.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEMomDev.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEScaledMomDev.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleESpher.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEtEtaPhi.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleEtThetaPhi.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleMCCart.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleMCMomDev.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleMCPInvSpher.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleMCSpher.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleMomDev.hh"
#include "TopQuarkAnalysis/TopKinFitter/src/TFitParticleSpher.hh"
// This linkdef file contains all "pragma link" for CLHEP inclusion
// into root including non-member operators and functions
// of Vector, Matrix, DiagMatrix and SymMatrix:
//#ifdef __CINT__
// ##################################################
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// ################## Functions #####################

#pragma link C++ class TAbsFitConstraint;
#pragma link C++ class TFitConstraintEp;
#pragma link C++ class TFitConstraintM;
#pragma link C++ class TFitConstraintMGaus;
//#pragma link C++ class TFitConstraintE;
#pragma link C++ class TAbsFitParticle;
#pragma link C++ class TFitParticleMomDev;
#pragma link C++ class TFitParticleCart;
#pragma link C++ class TFitParticleSpher;
#pragma link C++ class TFitParticleEMomDev;
#pragma link C++ class TFitParticleECart;
#pragma link C++ class TFitParticleESpher;
#pragma link C++ class TFitParticleMCMomDev;
#pragma link C++ class TFitParticleMCCart;
#pragma link C++ class TFitParticleMCSpher;
#pragma link C++ class TFitParticleEScaledMomDev;
#pragma link C++ class TFitParticleMCPInvSpher;
#pragma link C++ class TFitParticleEtEtaPhi;
#pragma link C++ class TFitParticleEtThetaPhi;
#pragma link C++ class TKinFitter;
#pragma link C++ class TSLToyGen;

//  #pragma link C++ class std::vector<TAbsFitParticle*>+;
//  #pragma link C++ class std::vector<TAbsFitConstraint*>+;

//#endif
