#ifndef EcalSimHitsValidation_H
#define EcalSimHitsValidation_H

/*
 * \file EcalSimHitsValidation.h
 *
 * \author C.Rovelli
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

class EcalSimHitsValidation : public edm::EDAnalyzer {
  typedef std::map<uint32_t, float, std::less<uint32_t>> MapType;

public:
  /// Constructor
  EcalSimHitsValidation(const edm::ParameterSet &ps);

  /// Destructor
  ~EcalSimHitsValidation() override;

protected:
  /// Analyze
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  // BeginJob
  void beginJob() override;

  // EndJob
  void endJob(void) override;

private:
  std::string g4InfoLabel;
  edm::EDGetTokenT<edm::HepMCProduct> HepMCToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> EBHitsCollectionToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> EEHitsCollectionToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> ESHitsCollectionToken;

  bool verbose_;

  DQMStore *dbe_;

  std::string outputFile_;

  MonitorElement *meGunEnergy_;
  MonitorElement *meGunEta_;
  MonitorElement *meGunPhi_;

  MonitorElement *meEBEnergyFraction_;
  MonitorElement *meEEEnergyFraction_;
  MonitorElement *meESEnergyFraction_;
};

#endif
