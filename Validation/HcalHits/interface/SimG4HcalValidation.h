///////////////////////////////////////////////////////////////////////////////
// File: SimG4HcalValidation.h
// Analysis and validation of simhits of HCal inside the CMSSW framework
///////////////////////////////////////////////////////////////////////////////
#ifndef Validation_HcalHits_SimG4HcalValidation_H
#define Validation_HcalHits_SimG4HcalValidation_H

#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include "Validation/HcalHits/interface/SimG4HcalHitCluster.h"
#include "Validation/HcalHits/interface/SimG4HcalHitJetFinder.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"

#include "SimDataFormats/CaloHit/interface/CaloHit.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

#include <iostream>
#include <memory>
#include <vector>
#include <string>

class G4Step;
class BeginOfJob;
class BeginOfRun;
class BeginOfEvent;
class EndOfEvent;

class SimG4HcalValidation : public SimProducer,
			    public Observer<const BeginOfJob *>, 
			    public Observer<const BeginOfRun *>, 
			    public Observer<const BeginOfEvent *>, 
			    public Observer<const EndOfEvent *>, 
			    public Observer<const G4Step *> {

public:
  SimG4HcalValidation(const edm::ParameterSet &p);
  virtual ~SimG4HcalValidation();

  void produce(edm::Event&, const edm::EventSetup&);

private:
  SimG4HcalValidation(const SimG4HcalValidation&); // stop default
  const SimG4HcalValidation& operator=(const SimG4HcalValidation&);

  void  init();

  // observer classes
  void update(const BeginOfJob * job);
  void update(const BeginOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const G4Step * step);
  void update(const EndOfEvent * evt);

  // jetfinding and analysis-related stuff
  void   fill(const EndOfEvent * ev);
  void   layerAnalysis(PHcalValidInfoLayer&);
  void   nxNAnalysis(PHcalValidInfoNxN&);
  void   jetAnalysis(PHcalValidInfoJets&);
  void   fetchHits(PHcalValidInfoLayer&);
  void   clear();
  void   collectEnergyRdir(const double, const double); 
  double getHcalScale(std::string, int) const; 


private:
  //Keep parameters to instantiate Jet finder later 
  SimG4HcalHitJetFinder *   jetf;

  //Keep reference to instantiate HcalNumberingFromDDD later
  HcalNumberingFromDDD *    numberingFromDDD;

  //Keep parameters to instantiate HcalTestNumberingScheme later
  HcalTestNumberingScheme * org;

  // Hit cache for cluster analysis
  std::vector<CaloHit>      hitcache;   // e, eta, phi, time, layer, calo type 

  // scale factors :
  std::vector<float>        scaleHB;
  std::vector<float>        scaleHE;
  std::vector<float>        scaleHF;
  
  // to read from parameter set
  std::vector<std::string>  names;
  double                    coneSize, ehitThreshold, hhitThreshold;
  float                     timeLowlim, timeUplim, eta0, phi0, jetThreshold; 
  bool                      applySampling, hcalOnly;
  int                       infolevel;
  std::string               labelLayer, labelNxN, labelJets;

  // eta and phi size of windows around eta0, phi0
  std::vector<double>       dEta;
  std::vector<double>       dPhi;

  // some private members for ananlysis 
  unsigned int              count;                  
  double                    edepEB, edepEE, edepHB, edepHE, edepHO;
  double                    edepd[5], edepl[20];
  double                    een, hen, hoen; // Energy sum in ECAL, HCAL, HO 
  double                    vhitec, vhithc, enEcal, enHcal;
};

#endif
