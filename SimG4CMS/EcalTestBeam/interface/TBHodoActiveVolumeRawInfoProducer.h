#ifndef TBHodoActiveVolumeRawInfoProducer_H
#define TBHodoActiveVolumeRawInfoProducer_H
/*
 * \file TBHodoActiveVolumeRawInfoProducer.h
 *
 * $Id: TBHodoActiveVolumeRawInfoProducer.h,v 1.4 2011/05/20 17:17:34 wmtan Exp $
 *
 */


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopePlaneRawHits.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"
#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"
#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

class TBHodoActiveVolumeRawInfoProducer: public edm::EDProducer{

  
 public:
  
  /// Constructor
  TBHodoActiveVolumeRawInfoProducer(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~TBHodoActiveVolumeRawInfoProducer();
  
  /// Produce digis out of raw data
  void produce(edm::Event & event, const edm::EventSetup& eventSetup);
  
private:

  double myThreshold;

  EcalTBHodoscopeGeometry * theTBHodoGeom_;
};

#endif
