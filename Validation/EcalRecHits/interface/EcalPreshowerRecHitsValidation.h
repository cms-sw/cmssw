#ifndef EcalPreshowerRecHitsValidation_H
#define EcalPreshowerRecHitsValidation_H

/*
 * \file EcalPreshowerRecHitsValidation.h
 *
 * \author C. Rovelli
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

#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

class EcalPreshowerRecHitsValidation : public DQMEDAnalyzer {
public:
  /// Constructor
  EcalPreshowerRecHitsValidation(const edm::ParameterSet &ps);

  /// Destructor
  ~EcalPreshowerRecHitsValidation() override;

protected:
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  /// Analyze
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  bool verbose_;

  edm::EDGetTokenT<EEUncalibratedRecHitCollection> EEuncalibrechitCollection_token_;
  edm::EDGetTokenT<EERecHitCollection> EErechitCollection_token_;
  edm::EDGetTokenT<ESRecHitCollection> ESrechitCollection_token_;

  MonitorElement *meESRecHitsEnergy_;
  MonitorElement *meESRecHitsEnergy_zp1st_;
  MonitorElement *meESRecHitsEnergy_zp2nd_;
  MonitorElement *meESRecHitsEnergy_zm1st_;
  MonitorElement *meESRecHitsEnergy_zm2nd_;
  MonitorElement *meESRecHitsMultip_;
  MonitorElement *meESRecHitsMultip_zp1st_;
  MonitorElement *meESRecHitsMultip_zp2nd_;
  MonitorElement *meESRecHitsMultip_zm1st_;
  MonitorElement *meESRecHitsMultip_zm2nd_;
  MonitorElement *meESEERecHitsEnergy_zp_;
  MonitorElement *meESEERecHitsEnergy_zm_;
  MonitorElement *meESRecHitsStripOccupancy_zp1st_[36];
  MonitorElement *meESRecHitsStripOccupancy_zm1st_[36];
  MonitorElement *meESRecHitsStripOccupancy_zp2nd_[36];
  MonitorElement *meESRecHitsStripOccupancy_zm2nd_[36];
};

#endif
