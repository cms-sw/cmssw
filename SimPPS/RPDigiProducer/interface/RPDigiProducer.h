#ifndef SimPPS_RPDigiProducer_RPDigiProducer_h
#define SimPPS_RPDigiProducer_RPDigiProducer_h

// -*- C++ -*-
//
// Package:    RPDigiProducer
// Class:      RPDigiProducer
//
#include "boost/shared_ptr.hpp"

// system include files
#include <memory>
#include <vector>
#include <map>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"

#include "SimPPS/RPDigiProducer/interface/RPDetDigitizer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "SimPPS/RPDigiProducer/interface/DeadChannelsManager.h"

//
// class decleration
//


namespace CLHEP {
  class HepRandomEngine;
}


class RPDigiProducer : public edm::EDProducer {
   public:
      explicit RPDigiProducer(const edm::ParameterSet&);
      ~RPDigiProducer() override;

   private:
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      void produce(edm::Event&, const edm::EventSetup&) override;
      void endJob() override;

      edm::DetSet<TotemRPDigi> convertRPStripDetSet(const edm::DetSet<TotemRPDigi>&);

    // ----------member data ---------------------------
      std::vector<std::string> RP_hit_containers_;
      typedef std::map<unsigned int, std::vector<PSimHit> > simhit_map;
      typedef simhit_map::iterator simhit_map_iterator;
      simhit_map SimHitMap;
      
      edm::ParameterSet conf_;
      std::map<RPDetId, boost::shared_ptr<RPDetDigitizer> > theAlgoMap;
      std::vector<edm::DetSet<TotemRPDigi> > theDigiVector;

      CLHEP::HepRandomEngine* rndEngine = nullptr;
      int verbosity_;

      /**
       * this variable answers the question whether given channel is dead or not
       */
      DeadChannelsManager deadChannelsManager;
      /**
       * this variable indicates whether we take into account dead channels or simulate as if all
       * channels work ok (by default we do not simulate dead channels)
       */
      bool simulateDeadChannels;

      edm::EDGetTokenT<CrossingFrame<PSimHit>> tokenCrossingFrameTotemRP;
};


#endif  //SimCTPPS_RPDigiProducer_RPDigiProducer_h
