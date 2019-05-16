#ifndef EcalRecHitsValidation_H
#define EcalRecHitsValidation_H

/*
 * \file EcalRecHitsValidation.h
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

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

class EcalRecHitsValidation : public DQMEDAnalyzer {
  typedef std::map<uint32_t, float, std::less<uint32_t>> MapType;

public:
  /// Constructor
  EcalRecHitsValidation(const edm::ParameterSet &ps);

  /// Destructor
  ~EcalRecHitsValidation() override;

protected:
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  /// Analyze
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  uint32_t getUnitWithMaxEnergy(MapType &themap);
  void findBarrelMatrix(int nCellInEta, int nCellInPhi, int CentralEta, int CentralPhi, int CentralZ, MapType &themap);
  void findEndcapMatrix(int nCellInX, int nCellInY, int CentralX, int CentralY, int CentralZ, MapType &themap);

private:
  std::string HepMCLabel;
  std::string hitsProducer_;

  bool verbose_;

  std::string outputFile_;

  edm::InputTag EBrechitCollection_;
  edm::InputTag EErechitCollection_;
  edm::InputTag ESrechitCollection_;
  edm::InputTag EBuncalibrechitCollection_;
  edm::InputTag EEuncalibrechitCollection_;
  // fix for consumes
  edm::EDGetTokenT<edm::HepMCProduct> HepMCLabel_Token_;
  edm::EDGetTokenT<EBRecHitCollection> EBrechitCollection_Token_;
  edm::EDGetTokenT<EERecHitCollection> EErechitCollection_Token_;
  edm::EDGetTokenT<ESRecHitCollection> ESrechitCollection_Token_;
  edm::EDGetTokenT<EBUncalibratedRecHitCollection> EBuncalibrechitCollection_Token_;
  edm::EDGetTokenT<EEUncalibratedRecHitCollection> EEuncalibrechitCollection_Token_;
  edm::EDGetTokenT<CrossingFrame<PCaloHit>> EBHits_Token_;
  edm::EDGetTokenT<CrossingFrame<PCaloHit>> EEHits_Token_;
  edm::EDGetTokenT<CrossingFrame<PCaloHit>> ESHits_Token_;

  MonitorElement *meGunEnergy_;
  MonitorElement *meGunEta_;
  MonitorElement *meGunPhi_;
  MonitorElement *meEBRecHitSimHitRatio_;
  MonitorElement *meEERecHitSimHitRatio_;
  MonitorElement *meESRecHitSimHitRatio_;
  MonitorElement *meEBRecHitSimHitRatio1011_;
  MonitorElement *meEERecHitSimHitRatio1011_;
  MonitorElement *meEBRecHitSimHitRatio12_;
  MonitorElement *meEERecHitSimHitRatio12_;
  MonitorElement *meEBRecHitSimHitRatio13_;
  MonitorElement *meEERecHitSimHitRatio13_;
  MonitorElement *meEBRecHitSimHitRatioGt35_;
  MonitorElement *meEERecHitSimHitRatioGt35_;
  MonitorElement *meEBUnRecHitSimHitRatio_;
  MonitorElement *meEEUnRecHitSimHitRatio_;
  MonitorElement *meEBUnRecHitSimHitRatioGt35_;
  MonitorElement *meEEUnRecHitSimHitRatioGt35_;
  MonitorElement *meEBe5x5_;
  MonitorElement *meEBe5x5OverSimHits_;
  MonitorElement *meEBe5x5OverGun_;
  MonitorElement *meEEe5x5_;
  MonitorElement *meEEe5x5OverSimHits_;
  MonitorElement *meEEe5x5OverGun_;

  MonitorElement *meEBRecHitLog10Energy_;
  MonitorElement *meEERecHitLog10Energy_;
  MonitorElement *meESRecHitLog10Energy_;
  MonitorElement *meEBRecHitLog10EnergyContr_;
  MonitorElement *meEERecHitLog10EnergyContr_;
  MonitorElement *meESRecHitLog10EnergyContr_;
  MonitorElement *meEBRecHitLog10Energy5x5Contr_;
  MonitorElement *meEERecHitLog10Energy5x5Contr_;

  MonitorElement *meEBRecHitsOccupancyFlag5_6_;
  MonitorElement *meEBRecHitsOccupancyFlag8_9_;

  MonitorElement *meEERecHitsOccupancyPlusFlag5_6_;
  MonitorElement *meEERecHitsOccupancyMinusFlag5_6_;
  MonitorElement *meEERecHitsOccupancyPlusFlag8_9_;
  MonitorElement *meEERecHitsOccupancyMinusFlag8_9_;

  MonitorElement *meEBRecHitFlags_;
  MonitorElement *meEBRecHitSimHitvsSimHitFlag5_6_;
  MonitorElement *meEBRecHitSimHitFlag6_;
  MonitorElement *meEBRecHitSimHitFlag7_;
  MonitorElement *meEB5x5RecHitSimHitvsSimHitFlag8_;

  MonitorElement *meEERecHitFlags_;
  MonitorElement *meEERecHitSimHitvsSimHitFlag5_6_;
  MonitorElement *meEERecHitSimHitFlag6_;
  MonitorElement *meEERecHitSimHitFlag7_;

  std::vector<uint32_t> crystalMatrix;
};

#endif
