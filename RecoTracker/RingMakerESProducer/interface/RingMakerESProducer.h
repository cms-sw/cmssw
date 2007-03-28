#ifndef RECOTRACKER_RINGMAKERESPRODUCER_H
#define RECOTRACKER_RINGMAKERESPRODUCER_H

//
// Package:         RecoTracker/RingMakerESProducer
// Class:           RingMakerESProducer
// 
// Description:     Uses the RingMaker object to construct
//                  and provide a Rings object.
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Tue Oct  3 23:51:34 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/02/05 19:09:03 $
// $Revision: 1.1 $
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/RingRecord/interface/RingRecord.h"

#include "RecoTracker/RingRecord/interface/Rings.h"

class RingMakerESProducer : public edm::ESProducer {

 public:

  RingMakerESProducer(const edm::ParameterSet&);
  ~RingMakerESProducer();

  typedef std::auto_ptr<Rings> ReturnType;

  ReturnType produce(const RingRecord&);

 private:

  bool        writeOut_;
  std::string fileName_;
  bool        dumpDetIds_;
  std::string detIdsDumpFileName_;
  std::string configuration_;
  Rings       *rings_;
};

#endif
