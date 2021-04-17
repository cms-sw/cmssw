#ifndef Validation_HcalHitValidation_H
#define Validation_HcalHitValidation_H

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class HcalHitValidation : public DQMEDAnalyzer {
public:
  HcalHitValidation(const edm::ParameterSet &ps);
  ~HcalHitValidation();

protected:
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  void analyzeHits(std::vector<PCaloHit> &);
  void analyzeLayer(edm::Handle<PHcalValidInfoLayer> &);
  void analyzeNxN(edm::Handle<PHcalValidInfoNxN> &);
  void analyzeJets(edm::Handle<PHcalValidInfoJets> &);

private:
  std::string g4Label, hcalHits, layerInfo, nxNInfo, jetsInfo;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hh_;
  edm::EDGetTokenT<PHcalValidInfoLayer> tok_iL_;
  edm::EDGetTokenT<PHcalValidInfoNxN> tok_iN_;
  edm::EDGetTokenT<PHcalValidInfoJets> tok_iJ_;
  std::string outFile_;
  bool verbose_, scheme_;
  bool checkHit_, checkLay_, checkNxN_, checkJet_;

  MonitorElement *meAllNHit_, *meBadDetHit_, *meBadSubHit_, *meBadIdHit_;
  MonitorElement *meHBNHit_, *meHENHit_, *meHONHit_, *meHFNHit_;
  MonitorElement *meDetectHit_, *meSubdetHit_, *meDepthHit_, *meEtaHit_;
  MonitorElement *mePhiHit_, *meEnergyHit_, *meTimeHit_, *meTimeWHit_;
  MonitorElement *meHBDepHit_, *meHEDepHit_, *meHODepHit_, *meHFDepHit_;
  MonitorElement *meHBEtaHit_, *meHEEtaHit_, *meHOEtaHit_, *meHFEtaHit_;
  MonitorElement *meHBPhiHit_, *meHEPhiHit_, *meHOPhiHit_, *meHFPhiHit_;
  MonitorElement *meHBEneHit_, *meHEEneHit_, *meHOEneHit_, *meHFEneHit_;
  MonitorElement *meHBTimHit_, *meHETimHit_, *meHOTimHit_, *meHFTimHit_;
  MonitorElement *mePMTHit_, *mePMTDepHit_, *mePMTEtaHit_, *mePMTPhiHit_;
  MonitorElement *mePMTEn1Hit_, *mePMTEn2Hit_, *mePMTTimHit_;

  static const int nLayersMAX = 20, nDepthsMAX = 5;
  MonitorElement *meLayerLay_, *meEtaHLay_, *mePhiHLay_, *meEneHLay_;
  MonitorElement *meDepHlay_, *meTimHLay_, *meTimWLay_;
  MonitorElement *meEtaPhi_, *meHitELay_, *meHitHLay_, *meHitTLay_;
  MonitorElement *meEneLLay_, *meEneLay_[nLayersMAX], *meLngLay_;
  MonitorElement *meEneDLay_, *meDepLay_[nDepthsMAX], *meEtotLay_;
  MonitorElement *meEHOLay_, *meEHBHELay_, *meEFibLLay_, *meEFibSLay_;
  MonitorElement *meEHFEmLay_, *meEHFHdLay_;

  MonitorElement *meEcalRNxN_, *meHcalRNxN_, *meHoRNxN_, *meEtotRNxN_;
  MonitorElement *meEcalNxN_, *meHcalNxN_, *meHoNxN_, *meEtotNxN_;
  MonitorElement *meEiNxN_, *meTiNxN_, *meTrNxN_;

  MonitorElement *meRJet_, *meTJet_, *meEJet_;
  MonitorElement *meEcalJet_, *meHcalJet_, *meHoJet_, *meEtotJet_, *meEcHcJet_;
  MonitorElement *meDetaJet_, *meDphiJet_, *meDrJet_, *meMassJet_;
  MonitorElement *meEneJet_, *meEtaJet_, *mePhiJet_;
};

#endif
