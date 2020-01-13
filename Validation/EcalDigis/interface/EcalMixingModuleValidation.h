#ifndef EcalMixingModuleValidation_H
#define EcalMixingModuleValidation_H

/*
 * \file EcalMixingModuleValidation.h
 *
 * \author F. Cossutti
 *
*/

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
//#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"

#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

namespace edm {
  class StreamID;
}

namespace CLHEP {
  class HepRandomEngine;
}

class EcalMixingModuleValidation : public DQMOneEDAnalyzer<> {
  typedef std::map<uint32_t, float, std::less<uint32_t> > MapType;

public:
  /// Constructor
  EcalMixingModuleValidation(const edm::ParameterSet& ps);

  /// Destructor
  ~EcalMixingModuleValidation() override;

  void bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) override;

protected:
  /// Analyze
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;

  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

  // EndRun
  void dqmEndRun(const edm::Run& r, const edm::EventSetup& c) override;

private:
  void checkPedestals(const edm::EventSetup& c);

  void findPedestal(const DetId& detId, int gainId, double& ped) const;

  void checkCalibrations(edm::EventSetup const& c);

  bool verbose_;

  std::string outputFile_;

  edm::EDGetTokenT<edm::HepMCProduct> HepMCToken_;

  edm::EDGetTokenT<EBDigiCollection> EBdigiCollectionToken_;
  edm::EDGetTokenT<EEDigiCollection> EEdigiCollectionToken_;
  edm::EDGetTokenT<ESDigiCollection> ESdigiCollectionToken_;

  edm::EDGetTokenT<CrossingFrame<PCaloHit> > crossingFramePCaloHitEBToken_, crossingFramePCaloHitEEToken_,
      crossingFramePCaloHitESToken_;

  std::map<int, double, std::less<int> > gainConv_;

  double barrelADCtoGeV_;
  double endcapADCtoGeV_;

  MonitorElement* meEBDigiMixRatiogt100ADC_;
  MonitorElement* meEEDigiMixRatiogt100ADC_;

  MonitorElement* meEBDigiMixRatioOriggt50pc_;
  MonitorElement* meEEDigiMixRatioOriggt40pc_;

  MonitorElement* meEBbunchCrossing_;
  MonitorElement* meEEbunchCrossing_;
  MonitorElement* meESbunchCrossing_;

  static const int nBunch = 21;

  MonitorElement* meEBBunchShape_[nBunch];
  MonitorElement* meEEBunchShape_[nBunch];
  MonitorElement* meESBunchShape_[nBunch];

  MonitorElement* meEBShape_;
  MonitorElement* meEEShape_;
  MonitorElement* meESShape_;

  MonitorElement* meEBShapeRatio_;
  MonitorElement* meEEShapeRatio_;
  MonitorElement* meESShapeRatio_;

  const EcalSimParameterMap* theParameterMap;
  //const CaloVShape * theEcalShape;
  ESShape* theESShape;
  EBShape* theEBShape;
  EEShape* theEEShape;

  //CaloHitResponse * theEcalResponse;
  CaloHitResponse* theESResponse;
  CaloHitResponse* theEBResponse;
  CaloHitResponse* theEEResponse;

  void computeSDBunchDigi(const edm::EventSetup& eventSetup,
                          const MixCollection<PCaloHit>& theHits,
                          MapType& ebSignalSimMap,
                          const EcalSubdetector& thisDet,
                          const double& theSimThreshold,
                          CLHEP::HepRandomEngine*);

  void bunchSumTest(std::vector<MonitorElement*>& theBunches,
                    MonitorElement*& theTotal,
                    MonitorElement*& theRatio,
                    int nSample);

  CLHEP::HepRandomEngine* randomEngine(edm::StreamID const& streamID);

  double esBaseline_;
  double esADCtokeV_;
  double esThreshold_;

  int theMinBunch;
  int theMaxBunch;

  const CaloGeometry* theGeometry;

  // the pedestals
  const EcalPedestals* thePedestals;

  int m_ESgain;
  const ESPedestals* m_ESpeds;
  const ESIntercalibConstants* m_ESmips;
  double m_ESeffwei;

  std::vector<CLHEP::HepRandomEngine*> randomEngines_;
};

#endif
