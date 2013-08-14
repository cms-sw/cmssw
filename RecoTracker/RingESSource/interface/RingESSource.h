#ifndef RECOTRACKER_RINGESSOURCE_H
#define RECOTRACKER_RINGESSOURCE_H

//
// Package:         RecoTracker/RingESSource
// Class:           RingESSource
// 
// Description:     Reads in ASCII dump of Rings
//                  and provides it to the event.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Thu Oct  5 01:35:14 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/03/15 20:16:47 $
// $Revision: 1.2 $
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/RingRecord/interface/Rings.h"
#include "RecoTracker/RingRecord/interface/RingRecord.h"

//
// class declaration
//

class RingESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {

 public:

  RingESSource(const edm::ParameterSet&);
  ~RingESSource();

  typedef std::auto_ptr<Rings> ReturnType;

  ReturnType produce(const RingRecord&);

 protected:

  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue &, edm::ValidityInterval &); 

 private:

  // ----------member data ---------------------------
  std::string fileName_;
  Rings *rings_;
  
};

#endif
