#ifndef SimG4CMS_HcalTestBeam_HcalTB06Analysis_H
#define SimG4CMS_HcalTestBeam_HcalTB06Analysis_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB06Analysis
//
/**\class HcalTB06Analysis HcalTB06Analysis.h SimG4CMS/HcalTestBeam/interface/HcalTB06Analysis.h
  
 Description: Analysis of 2004 Hcal Test beam simulation
  
 Usage: A Simwatcher class and can be activated from Oscarproducer module
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Oct 10 10:14:34 CEST 2006
//
  
// system include files
#include <iostream>
#include <memory>
#include <vector>
#include <string>
 
// user include files
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"
 
#include "SimG4CMS/HcalTestBeam/interface/HcalTB06Histo.h"

#include "SimDataFormats/CaloHit/interface/CaloHit.h"
#include "SimDataFormats/HcalTestBeam/interface/PHcalTB06Info.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"

#include <boost/cstdint.hpp>

class BeginOfRun;
class BeginOfEvent;
class EndOfEvent;

class HcalTB06Analysis : public SimProducer,
			 public Observer<const BeginOfRun *>,
			 public Observer<const BeginOfEvent *>,
			 public Observer<const EndOfEvent *>,
			 public Observer<const G4Step *> {

public:

  HcalTB06Analysis(const edm::ParameterSet &p);
  virtual ~HcalTB06Analysis();

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  HcalTB06Analysis(const HcalTB06Analysis&); // stop default
  const HcalTB06Analysis& operator=(const HcalTB06Analysis&);
 
  void  init();
 
  // observer methods
  void update(const BeginOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const G4Step * step);
  void update(const EndOfEvent * evt);

  //User methods
  void fillBuffer(const EndOfEvent * evt);
  void finalAnalysis();
  void fillEvent(PHcalTB06Info&);

  void   clear();

private:

  HcalTB06Histo*             histo;

  // to read from parameter set
  double                     beamOffset;
  int                        iceta, icphi;
  std::vector<std::string>   names;
  G4RotationMatrix*          beamline_RM;

  // Constants for the run
  int                        count;
    
  // Constants for the event
  int                        nPrimary, particleType;
  double                     pInit, etaInit, phiInit;
  std::vector<CaloHit>       ecalHitCache;
  std::vector<CaloHit>       hcalHitCache, hcalHitLayer;
  double                     etots, eecals, ehcals;

  bool                       pvFound;
  int                        evNum, pvType;
  G4ThreeVector              pvPosition, pvMomentum, pvUVW;
  std::vector<int>           secTrackID, secPartID;
  std::vector<G4ThreeVector> secMomentum;
  std::vector<double>        secEkin;
  std::vector<int>           shortLivedSecondaries;
};

#endif
