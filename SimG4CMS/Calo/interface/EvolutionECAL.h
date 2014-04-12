#ifndef CalibCalorimetry_EcalPlugins_EvolutionECAL_H
#define CalibCalorimetry_EcalPlugins_EvolutionECAL_H
// system include files

#include <vector>
#include <typeinfo>
#include <string>
#include <map>

#include <time.h>
#include <stdio.h>

#include <math.h>
#include "TF1.h"
#include "TH1F.h"
#include "TMath.h"
#include "TObjArray.h"
#include "TFile.h"
#include "TString.h"
#include <fstream>
#include <sstream>

class EvolutionECAL {

public:

  EvolutionECAL();
  virtual ~EvolutionECAL();


  double LightCollectionEfficiency(double z, double mu);
 double DamageProfileEta(double eta);
 double DamageProfileEtaAPD(double eta);
 double InducedAbsorptionHadronic(double lumi, double eta);
 double DoseLongitudinalProfile(double z);
 double InducedAbsorptionEM(double lumi, double eta);
 double DegradationMeanEM50GeV(double mu);
 double DegradationNonLinearityEM50GeV(double mu, double ene);
 double ResolutionConstantTermEM50GeV(double mu);
 double ChargeVPTCathode(double instLumi, double eta, double integralLumi);
 double AgingVPT(double instLumi, double integralLumi, double eta); 
 double NoiseFactorFE(double lumi, double eta);
 Double_t  EquilibriumFractionColorCentersEM(double *x, double *par);
 double LightCollectionEfficiencyWeighted(double z, double mu_ind); 



};



#endif

