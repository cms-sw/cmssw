#ifndef EcalTBMCInfoProducer_H
#define EcalTBMCInfoProducer_H
/*
 * \file EcalTBMCInfoProducer.h
 *
 * $Id: EcalTBMCInfoProducer.h,v 1.2 2006/07/18 14:09:05 fabiocos Exp $
 *
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "Geometry/EcalTestBeam/interface/EcalTBCrystalMap.h"

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"

#include <iostream>
#include <fstream>
#include <vector>

class EcalTBMCInfoProducer: public edm::EDProducer{
  
 public:
  
  /// Constructor
  EcalTBMCInfoProducer(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~EcalTBMCInfoProducer();
  
  /// Produce digis out of raw data
  void produce(edm::Event & event, const edm::EventSetup& eventSetup);
  
  // BeginJob
  //void beginJob(const edm::EventSetup& c);
  
  // EndJob
  //void endJob(void);
  

private:

  double beamEta;
  double beamPhi;
  double beamTheta;

  int crysNumber;

  double beamXoff;
  double beamYoff;

  double partXhodo;
  double partYhodo;

  EcalTBCrystalMap * theTestMap;

  HepRotation * fromCMStoTB;

  std::string GenVtxLabel;

};

#endif
