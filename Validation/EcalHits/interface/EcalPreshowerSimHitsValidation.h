#ifndef EcalPreshowerSimHitsValidation_H
#define EcalPreshowerSimHitsValidation_H

/*
 * \file EcalPreshowerSimHitsValidation.h
 *
 * \author C.Rovelli
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
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
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

class EcalPreshowerSimHitsValidation : public edm::EDAnalyzer {
  typedef std::map<uint32_t, float, std::less<uint32_t>> MapType;

public:
  /// Constructor
  EcalPreshowerSimHitsValidation(const edm::ParameterSet &ps);

  /// Destructor
  ~EcalPreshowerSimHitsValidation() override;

protected:
  /// Analyze
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  // BeginJob
  void beginJob() override;

  // EndJob
  void endJob(void) override;

private:
  std::string HepMCLabel;
  std::string g4InfoLabel;
  std::string EEHitsCollection;
  std::string ESHitsCollection;

  edm::EDGetTokenT<edm::HepMCProduct> HepMCToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> EEHitsToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> ESHitsToken;

  bool verbose_;

  DQMStore *dbe_;

  std::string outputFile_;

  MonitorElement *menESHits1zp_;
  MonitorElement *menESHits2zp_;

  MonitorElement *menESHits1zm_;
  MonitorElement *menESHits2zm_;

  MonitorElement *meEShitLog10Energy_;
  MonitorElement *meEShitLog10EnergyNorm_;

  MonitorElement *meESEnergyHits1zp_;
  MonitorElement *meESEnergyHits2zp_;

  MonitorElement *meESEnergyHits1zm_;
  MonitorElement *meESEnergyHits2zm_;

  MonitorElement *meE1alphaE2zp_;
  MonitorElement *meE1alphaE2zm_;

  MonitorElement *meEEoverESzp_;
  MonitorElement *meEEoverESzm_;

  MonitorElement *me2eszpOver1eszp_;
  MonitorElement *me2eszmOver1eszm_;
};

#endif
