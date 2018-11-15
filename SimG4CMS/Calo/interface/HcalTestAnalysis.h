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
  ~HcalTestAnalysis() override;

private:
  // observer classes
  void update(const BeginOfJob * run) override;
  void update(const BeginOfRun * run) override;
  void update(const BeginOfEvent * evt) override;
  void update(const EndOfEvent * evt) override;
  void update(const G4Step * step) override;

  // analysis-related stuff
  std::vector<int> layerGrouping(int);
  std::vector<int> towersToAdd(int centre, int nadd);
  void   fill(const EndOfEvent * ev);
  void   qieAnalysis(CLHEP::HepRandomEngine*);
  void   layerAnalysis();
  double timeOfFlight(int det, int layer, double eta);

private:

  //Keep parameters to instantiate HcalTestHistoClass later
  std::string                              fileName_;

  // Qie Analysis
  std::unique_ptr<HcalQie>                 myqie_;
  int                                      addTower_;

  // Private Tuples
  std::unique_ptr<HcalTestHistoManager>    tuplesManager_;
  HcalTestHistoClass*                      tuples_;

  // Numbering scheme
  std::unique_ptr<HcalNumberingFromDDD>    numberingFromDDD_;
  const HcalDDDSimConstants*               hcons_;
  HcalTestNumberingScheme*                 org_;

  // Hits for qie analysis
  std::vector<CaloHit>                     caloHitCache_; 
  std::vector<int>                         group_, tower_;
  int                                      nGroup_, nTower_;
  
  // to read from ParameterSet
  std::vector<std::string>                 names_;
  double                                   eta0_, phi0_;
  int                                      centralTower_;

  // some private members for ananlysis 
  unsigned int                             count_;                  
  double                                   edepEB_, edepEE_, edepHB_, edepHE_;
  double                                   edepHO_, edepl_[20];
  double                                   mudist_[20];   // Distance of muon from central part
};

#endif
