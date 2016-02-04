#ifndef SimG4CMS_HcalTestBeam_HcalTB02Analysis_H
#define SimG4CMS_HcalTestBeam_HcalTB02Analysis_H
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
// $Id: HcalTB02Analysis.h,v 1.2 2006/11/13 10:32:14 sunanda Exp $
//
  
// system include files
#include <iostream>
#include <memory>
#include <map>
#include <vector>
#include <string>
 
// user include files
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
 
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02Histo.h"
#include "SimDataFormats/HcalTestBeam/interface/HcalTB02HistoClass.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"

#include <boost/cstdint.hpp>

class HcalTB02Analysis : public SimProducer,
			 public Observer<const BeginOfEvent *>,
			 public Observer<const EndOfEvent *> {

public:

  HcalTB02Analysis(const edm::ParameterSet &p);
  virtual ~HcalTB02Analysis();

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  HcalTB02Analysis(const HcalTB02Analysis&); // stop default
  const HcalTB02Analysis& operator=(const HcalTB02Analysis&);
 
  // observer methods
  void update(const BeginOfEvent * evt);
  void update(const EndOfEvent * evt);

  void fillEvent(HcalTB02HistoClass&);
  void clear();
  void finish();
 
private:

  // Private Tuples
  HcalTB02Histo*                      histo;

  // to read from parameter set
  bool                                hcalOnly;
  std::string                         fileNameTuple;
  std::vector<std::string>            names;

  //To be saved
  std::map<int,float>                 energyInScints, energyInCrystals;
  std::map<int,float>                 primaries;
  int                                 particleType;
  double                              eta, phi, pInit, incidentEnergy;
  float                               SEnergy, E7x7Matrix, E5x5Matrix;
  float                               SEnergyN, E7x7MatrixN, E5x5MatrixN;
  int                                 maxTime;
  double                              xIncidentEnergy;
  float                               xSEnergy, xSEnergyN;
  float                               xE3x3Matrix, xE5x5Matrix;
  float                               xE3x3MatrixN, xE5x5MatrixN;
 };

#endif
