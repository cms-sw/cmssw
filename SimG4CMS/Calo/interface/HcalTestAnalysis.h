#ifndef SimG4CMS_HcalTestAnalysis_H
#define SimG4CMS_HcalTestAnalysis_H
///////////////////////////////////////////////////////////////////////////////
// File: HcalTestAnalysis.h
// Analysis of simhits inside the OSCAR framework
///////////////////////////////////////////////////////////////////////////////

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "SimDataFormats/CaloHit/interface/CaloHit.h"
#include "SimDataFormats/CaloTest/interface/HcalTestHistoClass.h"
#include "SimG4CMS/Calo/interface/HcalQie.h"
#include "SimG4CMS/Calo/interface/HcalTestHistoManager.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

#include <iostream>
#include <memory>
#include <vector>
#include <string>


class G4Step;
class BeginOfJob;
class BeginOfRun;
class BeginOfEvent;
class EndOfEvent;

namespace CLHEP {
  class HepRandomEngine;
}

class HcalTestAnalysis : public SimWatcher,
			 public Observer<const BeginOfJob *>, 
			 public Observer<const BeginOfRun *>, 
			 public Observer<const BeginOfEvent *>, 
			 public Observer<const EndOfEvent *>, 
			 public Observer<const G4Step *> {

public:
  HcalTestAnalysis(const edm::ParameterSet &p);
  virtual ~HcalTestAnalysis();

private:
  // observer classes
  void update(const BeginOfJob * run);
  void update(const BeginOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const EndOfEvent * evt);
  void update(const G4Step * step);

  // analysis-related stuff
  std::vector<int> layerGrouping(int);
  std::vector<int> towersToAdd(int centre, int nadd);
  void   fill(const EndOfEvent * ev);
  void   qieAnalysis(CLHEP::HepRandomEngine*);
  void   layerAnalysis();
  double timeOfFlight(int det, int layer, double eta);

private:

  //Keep parameters to instantiate HcalTestHistoClass later
  std::string               fileName;

  // Qie Analysis
  HcalQie *                 myqie;
  int                       addTower;

  // Private Tuples
  std::auto_ptr<HcalTestHistoManager>    tuplesManager;
  HcalTestHistoClass        *tuples;

  // Numbering scheme
  HcalNumberingFromDDD      *numberingFromDDD;
  HcalDDDSimConstants       *hcons;
  HcalTestNumberingScheme   *org;

  // Hits for qie analysis
  std::vector<CaloHit>      caloHitCache; 
  std::vector<int>          group_, tower_;
  int                       nGroup, nTower;
  
  // to read from ParameterSet
  std::vector<std::string>  names;
  double                    eta0, phi0;
  int                       centralTower;

  // some private members for ananlysis 
  unsigned int              count;                  
  double                    edepEB, edepEE, edepHB, edepHE, edepHO;
  double                    edepl[20];
  double                    mudist[20];   // Distance of muon from central part
};

#endif
