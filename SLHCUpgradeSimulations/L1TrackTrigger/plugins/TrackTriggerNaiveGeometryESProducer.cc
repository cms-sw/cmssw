// -*- C++ -*-
//
// Package:    TrackTriggerNaiveGeometryESProducer
// Class:      TrackTriggerNaiveGeometryESProducer
// 
/**\class TrackTriggerNaiveGeometryESProducer TrackTriggerNaiveGeometryESProducer.h SLHCUpgradeSimulations/L1Trigger/src/TrackTriggerNaiveGeometryESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu Mar 20 10:59:56 CET 2008
// $Id: TrackTriggerNaiveGeometryESProducer.cc,v 1.2 2010/02/03 09:46:37 arose Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackTriggerNaiveGeometry.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackTriggerNaiveGeometryRcd.h"

#include "TMath.h"

//
// class decleration
//

class TrackTriggerNaiveGeometryESProducer : public edm::ESProducer {
public:
  TrackTriggerNaiveGeometryESProducer(const edm::ParameterSet&);
  ~TrackTriggerNaiveGeometryESProducer();
  
  std::auto_ptr<TrackTriggerNaiveGeometry> produce(const TrackTriggerNaiveGeometryRcd& rcd);
private:

  // barrel (everything applies to a half-detector!)
  std::vector<double> radii_;
  std::vector<double> lengths_;
  std::vector<double> barrelTowZSize_;
  std::vector<double> barrelTowPhiSize_;
  std::vector<double> barrelPixelZSize_;
  std::vector<double> barrelPixelPhiSize_;

  // endcap (everything applies to a half-detector!)
//  std::vector<double> diskRadii_;
//  std::vector<double> diskZPos_;
//  std::vector<double> diskTowRBoundaries_;
//  std::vector<double> diskTowPhiBoundaries_;
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackTriggerNaiveGeometryESProducer::TrackTriggerNaiveGeometryESProducer(const edm::ParameterSet& iConfig)
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  
  // setup layers/disks and positions
  radii_ = iConfig.getParameter< std::vector<double> >("radii");
  lengths_ = iConfig.getParameter< std::vector<double> >("lengths");
  barrelTowZSize_ = iConfig.getParameter< std::vector<double> >("barrelTowZSize");
  barrelTowPhiSize_ = iConfig.getParameter< std::vector<double> >("barrelTowPhiSize");
  barrelPixelZSize_ = iConfig.getParameter< std::vector<double> >("barrelPixelZSize");
  barrelPixelPhiSize_ = iConfig.getParameter< std::vector<double> >("barrelPixelPhiSize");

  LogDebug("L1Tracks") << "ES parameters : N stations " << radii_.size() << std::endl;
//  LogDebug("L1Tracks") << "ES parameters : N disks " << diskRadii_.size() << std::endl;

}


TrackTriggerNaiveGeometryESProducer::~TrackTriggerNaiveGeometryESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
std::auto_ptr<TrackTriggerNaiveGeometry>
TrackTriggerNaiveGeometryESProducer::produce(const TrackTriggerNaiveGeometryRcd& rcd)
{
   using namespace edm::es;

   std::auto_ptr<TrackTriggerNaiveGeometry> 
     pGeom(new TrackTriggerNaiveGeometry(radii_, 
					 lengths_, 
					 barrelTowZSize_, 
					 barrelTowPhiSize_, 
					 barrelPixelZSize_.at(0), 
					 barrelPixelPhiSize_.at(0))
	   );
   
   return pGeom ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(TrackTriggerNaiveGeometryESProducer);

