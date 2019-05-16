#ifndef EcalBarrelSimHitsValidation_H
#define EcalBarrelSimHitsValidation_H

/*
 * \file EcalBarrelSimHitsValidation.h
 *
 * \author C.Rovelli
 *
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

class EcalBarrelSimHitsValidation : public edm::EDAnalyzer {
  typedef std::map<uint32_t, float, std::less<uint32_t>> MapType;

public:
  /// Constructor
  EcalBarrelSimHitsValidation(const edm::ParameterSet &ps);

  /// Destructor
  ~EcalBarrelSimHitsValidation() override;

protected:
  /// Analyze
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  // BeginJob
  void beginJob() override;

  // EndJob
  void endJob(void) override;

private:
  uint32_t getUnitWithMaxEnergy(MapType &themap);

  virtual float energyInMatrixEB(
      int nCellInEta, int nCellInPhi, int centralEta, int centralPhi, int centralZ, MapType &themap);

  std::vector<uint32_t> getIdsAroundMax(
      int nCellInEta, int nCellInPhi, int centralEta, int centralPhi, int centralZ, MapType &themap);

  bool fillEBMatrix(
      int nCellInEta, int nCellInPhi, int CentralEta, int CentralPhi, int CentralZ, MapType &fillmap, MapType &themap);

  float eCluster2x2(MapType &themap);
  float eCluster4x4(float e33, MapType &themap);

  std::string g4InfoLabel;
  std::string EBHitsCollection;
  std::string ValidationCollection;

  edm::EDGetTokenT<edm::PCaloHitContainer> EBHitsToken;
  edm::EDGetTokenT<PEcalValidInfo> ValidationCollectionToken;

  bool verbose_;

  DQMStore *dbe_;

  std::string outputFile_;

  int myEntries;
  float eRLength[26];

  MonitorElement *menEBHits_;

  MonitorElement *menEBCrystals_;

  MonitorElement *meEBOccupancy_;

  MonitorElement *meEBLongitudinalShower_;

  MonitorElement *meEBhitEnergy_;

  MonitorElement *meEBhitLog10Energy_;

  MonitorElement *meEBhitLog10EnergyNorm_;

  MonitorElement *meEBhitLog10Energy25Norm_;

  MonitorElement *meEBhitEnergy2_;

  MonitorElement *meEBcrystalEnergy_;

  MonitorElement *meEBcrystalEnergy2_;

  MonitorElement *meEBe1_;
  MonitorElement *meEBe4_;
  MonitorElement *meEBe9_;
  MonitorElement *meEBe16_;
  MonitorElement *meEBe25_;

  MonitorElement *meEBe1oe4_;
  MonitorElement *meEBe1oe9_;
  MonitorElement *meEBe4oe9_;
  MonitorElement *meEBe9oe16_;
  MonitorElement *meEBe1oe25_;
  MonitorElement *meEBe9oe25_;
  MonitorElement *meEBe16oe25_;
};

#endif
