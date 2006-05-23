#ifndef HcalTestBeam_HcalTB02Analysis_H
#define HcalTestBeam_HcalTB02Analysis_H
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02Analysis
//
/**\class HcalTB02Analysis HcalTB02Analysis.h SimG4CMS/HcalTestBeam/interface/HcalTB02Analysis.h
  
 Description: Analysis of 2004 Hcal Test beam simulation
  
 Usage: A Simwatcher class and can be activated from Oscarproducer module
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu May 18 10:14:34 CEST 2006
// $Id$
//
  
// system include files
#include <iostream>
#include <memory>
#include <vector>
#include <string>
 
// user include files
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
 
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02Histo.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02HistoManager.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02HistoClass.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"

#include <boost/cstdint.hpp>

class HcalTB02Analysis : public SimWatcher,
			 public Observer<const BeginOfJob *>,
			 public Observer<const BeginOfEvent *>,
			 public Observer<const EndOfEvent *> {

public:

  HcalTB02Analysis(const edm::ParameterSet &p);
  virtual ~HcalTB02Analysis();

private:

  HcalTB02Analysis(const HcalTB02Analysis&); // stop default
  const HcalTB02Analysis& operator=(const HcalTB02Analysis&);
 
  // observer methods
  void update(const BeginOfJob * job);
  void update(const BeginOfEvent * evt);
  void update(const EndOfEvent * evt);

  void  finish();
 
private:

  // Private Tuples
  std::auto_ptr<HcalTB02HistoManager> tuplesManager;
  HcalTB02HistoClass*                 tuples;
  HcalTB02Histo*                      histo;

  // to read from parameter set
  bool                                hcalOnly;
  std::string                         fileNameTuple;
  std::vector<std::string>            names;
 };

#endif
