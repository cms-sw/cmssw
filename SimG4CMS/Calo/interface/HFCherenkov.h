#ifndef SimG4CMS_HFCherenkov_h
#define SimG4CMS_HFCherenkov_h 1
///////////////////////////////////////////////////////////////////////////////
// File:  HFCherenkov.h
// Description: Generate Cherenkov photons for HF
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4DynamicParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"

#include <vector>

class HFCherenkov {
  
public:

   HFCherenkov(edm::ParameterSet const & p);
   virtual ~HFCherenkov();
  
   int                 computeNPE(G4Step* step, G4ParticleDefinition* pDef,
				  double pBeta, double u, double v, double w, 
				  double step_length, double zFiber, 
				  double Dose, int Npe_Dose);
   
   int                 computeNPEinPMT(G4ParticleDefinition* pDef,double pBeta,
                                       double u, double v, double w, 
                                       double step_length);

   int                 computeNPhTrapped(double pBeta, double u, double v, 
					 double w, double step_length,
					 double zFiber, double Dose,
					 int Npe_Dose);
   double              smearNPE(G4int Npe);				  

   std::vector<double> getMom();
   std::vector<double> getWL();
   std::vector<double> getWLIni();
   std::vector<double> getWLTrap();
   std::vector<double> getWLAtten();
   std::vector<double> getWLHEM();
   std::vector<double> getWLQEff();
   void                clearVectors();
					  
private:

   bool                isApplicable(const G4ParticleDefinition* aParticleType);
   // Returns true -> 'is applicable', for all charged particles.
   int                 computeNbOfPhotons(double pBeta, double step_length);
   double              computeQEff(double wavelength);
   double              computeHEMEff(double wavelength);

private:

   double              ref_index;
   double              lambda1, lambda2;
   double              aperture, aperturetrapped, apertureTrap;
   double              gain, fibreR, sinPsimax;
   bool                checkSurvive;
   bool                UseNewPMT;

   G4ThreeVector       phMom;
   std::vector<double> wl;
   std::vector<double> momZ;
   std::vector<double> wlini;
   std::vector<double> wltrap;
   std::vector<double> wlatten;
   std::vector<double> wlhem;
   std::vector<double> wlqeff;
};

#endif

