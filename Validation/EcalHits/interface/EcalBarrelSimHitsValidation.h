#ifndef EcalBarrelSimHitsValidation_H
#define EcalBarrelSimHitsValidation_H

/*
 * \file EcalBarrelSimHitsValidation.h
 *
 * \author C.Rovelli
 *
 */

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

#include <vector>
#include <map>

class MonitorElement;

class EcalBarrelSimHitsValidation : public DQMEDAnalyzer {

  typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

 public:

  /// Constructor
  EcalBarrelSimHitsValidation(edm::ParameterSet const&);

  /// Destructor
  ~EcalBarrelSimHitsValidation();

 protected:

  /// Analyze
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

 private:

  uint32_t getUnitWithMaxEnergy(MapType& themap);

  virtual float energyInMatrixEB(int nCellInEta, int nCellInPhi, 
                                 int centralEta, int centralPhi, int centralZ,
                                 MapType& themap); 

  std::vector<uint32_t> getIdsAroundMax(int nCellInEta, int nCellInPhi, 
                                        int centralEta, int centralPhi, int centralZ,
                                        MapType& themap); 

 
  bool  fillEBMatrix(int nCellInEta, int nCellInPhi,
                     int CentralEta, int CentralPhi,int CentralZ,
                     MapType& fillmap, MapType&  themap);
 
  float eCluster2x2( MapType& themap);
  float eCluster4x4(float e33,MapType& themap);

  edm::EDGetTokenT<edm::PCaloHitContainer> EBHitsToken;
  edm::EDGetTokenT<PEcalValidInfo> ValidationCollectionToken;

  int myEntries;
  float eRLength[26];

  MonitorElement* menEBHits_;

  MonitorElement* menEBCrystals_;

  MonitorElement* meEBOccupancy_;

  MonitorElement* meEBLongitudinalShower_;

  MonitorElement* meEBhitEnergy_;

  MonitorElement* meEBhitLog10Energy_;

  MonitorElement* meEBhitLog10EnergyNorm_;

  MonitorElement* meEBhitLog10Energy25Norm_;

  MonitorElement* meEBhitEnergy2_;

  MonitorElement* meEBcrystalEnergy_;

  MonitorElement* meEBcrystalEnergy2_;

  MonitorElement* meEBe1_; 
  MonitorElement* meEBe4_; 
  MonitorElement* meEBe9_; 
  MonitorElement* meEBe16_; 
  MonitorElement* meEBe25_; 

  MonitorElement* meEBe1oe4_;
  MonitorElement* meEBe1oe9_;
  MonitorElement* meEBe4oe9_;
  MonitorElement* meEBe9oe16_;
  MonitorElement* meEBe1oe25_;
  MonitorElement* meEBe9oe25_; 
  MonitorElement* meEBe16oe25_;
};

#endif
