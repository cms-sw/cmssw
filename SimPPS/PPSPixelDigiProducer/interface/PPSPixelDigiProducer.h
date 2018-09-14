
#ifndef SimPPS_PPSPixelDigiProducer_PPSPixelDigiProducer_h
#define SimPPS_PPSPixelDigiProducer_PPSPixelDigiProducer_h

// -*- C++ -*-
//
// Package:    PPSPixelDigiProducer
// Class:      CTPPSPixelDigiProducer
// 
/**\class CTPPSPixelDigiProducer PPSPixelDigiProducer.cc SimPPS/PPSPixelDigiProducer/plugins/PPSPixelDigiProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  F.Ferro
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
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

//  ****  CTPPS
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigiCollection.h"

#include "SimPPS/PPSPixelDigiProducer/interface/RPixDetDigitizer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DataFormats/Common/interface/DetSet.h"

// DB
#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSPixelGainCalibrations.h"
#include "RecoCTPPS/PixelLocal/interface/CTPPSPixelGainCalibrationDBService.h"


namespace CLHEP {
  class HepRandomEngine;
}

class CTPPSPixelDigiProducer : public edm::EDProducer {
   public:
      explicit CTPPSPixelDigiProducer(const edm::ParameterSet&);
      ~CTPPSPixelDigiProducer() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
   private:
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      void produce(edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
      
      // ----------member data ---------------------------
      std::vector<std::string> RPix_hit_containers_;
      typedef std::map<unsigned int, std::vector<PSimHit> > simhit_map;
      typedef simhit_map::iterator simhit_map_iterator;
      simhit_map SimHitMap;
      
      edm::ParameterSet conf_;

      std::map<uint32_t, boost::shared_ptr<RPixDetDigitizer> > theAlgoMap;  //DetId = uint32_t 

      std::vector<edm::DetSet<CTPPSPixelDigi> > theDigiVector;

      CLHEP::HepRandomEngine* rndEngine = nullptr;
      int verbosity_;

      CTPPSPixelGainCalibrationDBService theGainCalibrationDB;

      edm::EDGetTokenT<CrossingFrame<PSimHit>> tokenCrossingFramePPSPixel;
};

#endif
