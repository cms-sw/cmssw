#ifndef EcalEndcapSimHitsValidation_H
#define EcalEndcapSimHitsValidation_H

/*
 * \file EcalEndcapSimHitsValidation.h
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

class EcalEndcapSimHitsValidation : public DQMEDAnalyzer {

  typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

 public:

  /// Constructor
  EcalEndcapSimHitsValidation(edm::ParameterSet const&);

  /// Destructor
  ~EcalEndcapSimHitsValidation();

 protected:

  /// Analyze
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

 private:

  uint32_t getUnitWithMaxEnergy(MapType& themap);

  virtual float energyInMatrixEE(int nCellInX, int nCellInY, 
                                 int centralX, int centralY, int centralZ,
                                 MapType& themap); 
 
  std::vector<uint32_t> getIdsAroundMax(int nCellInX, int nCellInY, 
                                        int centralX, int centralY, int centralZ,
                                        MapType& themap); 

  bool  fillEEMatrix(int nCellInX, int nCellInY,
                     int CentralX, int CentralY,int CentralZ,
                     MapType& fillmap, MapType&  themap);

  float eCluster2x2( MapType& themap);
  float eCluster4x4(float e33,MapType& themap);

  edm::EDGetTokenT<edm::PCaloHitContainer> EEHitsToken;
  edm::EDGetTokenT<PEcalValidInfo> ValidationCollectionToken;

  int myEntries;
  float eRLength[26];

  MonitorElement* meEEzpHits_;
  MonitorElement* meEEzmHits_;

  MonitorElement* meEEzpCrystals_;
  MonitorElement* meEEzmCrystals_;

  MonitorElement* meEEzpOccupancy_;
  MonitorElement* meEEzmOccupancy_;

  MonitorElement* meEELongitudinalShower_;

  MonitorElement* meEEHitEnergy_;

  MonitorElement* meEEhitLog10Energy_;

  MonitorElement* meEEhitLog10EnergyNorm_;

  MonitorElement* meEEhitLog10Energy25Norm_;

  MonitorElement* meEEHitEnergy2_;

  MonitorElement* meEEcrystalEnergy_;
  MonitorElement* meEEcrystalEnergy2_;

  MonitorElement* meEEe1_; 
  MonitorElement* meEEe4_; 
  MonitorElement* meEEe9_; 
  MonitorElement* meEEe16_; 
  MonitorElement* meEEe25_; 

  MonitorElement* meEEe1oe4_;
  MonitorElement* meEEe1oe9_;
  MonitorElement* meEEe4oe9_;
  MonitorElement* meEEe9oe16_;
  MonitorElement* meEEe1oe25_;
  MonitorElement* meEEe9oe25_;
  MonitorElement* meEEe16oe25_;
};

#endif
