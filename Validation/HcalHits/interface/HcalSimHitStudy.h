#ifndef SimG4CMS_HcalSimHitStudy_H
#define SimG4CMS_HcalSimHitStudy_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class HcalSimHitStudy : public DQMEDAnalyzer {
public:
  HcalSimHitStudy(const edm::ParameterSet &ps);
  ~HcalSimHitStudy() override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

protected:
  // void endJob   ();
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  void analyzeHits(std::vector<PCaloHit> &);

private:
  const HcalDDDRecConstants *hcons_;
  int maxDepthHB_, maxDepthHE_;
  int maxDepthHO_, maxDepthHF_;
  int maxDepth_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_HRNDC_;

  int iphi_bins;
  float iphi_min, iphi_max;
  int ieta_bins_HB;
  float ieta_min_HB, ieta_max_HB;
  int ieta_bins_HE;
  float ieta_min_HE, ieta_max_HE;
  int ieta_bins_HO;
  float ieta_min_HO, ieta_max_HO;
  int ieta_bins_HF;
  float ieta_min_HF, ieta_max_HF;

  std::string g4Label, hcalHits, outFile_;
  bool verbose_, checkHit_, testNumber_, hep17_;

  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;

  MonitorElement *meAllNHit_, *meBadDetHit_, *meBadSubHit_, *meBadIdHit_;
  MonitorElement *meHBNHit_, *meHENHit_, *meHONHit_, *meHFNHit_;
  MonitorElement *meDetectHit_, *meSubdetHit_, *meDepthHit_, *meEtaHit_, *meEtaPhiHit_;
  std::vector<MonitorElement *> meEtaPhiHitDepth_;
  MonitorElement *mePhiHit_, *mePhiHitb_, *meEnergyHit_, *meTimeHit_, *meTimeWHit_;
  MonitorElement *meHBDepHit_, *meHEDepHit_, *meHODepHit_, *meHFDepHit_, *meHFDepHitw_;
  MonitorElement *meHBEtaHit_, *meHEEtaHit_, *meHOEtaHit_, *meHFEtaHit_;
  MonitorElement *meHBPhiHit_, *meHEPhiHit_, *meHOPhiHit_, *meHFPhiHit_;
  MonitorElement *meHBEneHit_, *meHEEneHit_, *meHOEneHit_, *meHFEneHit_;
  MonitorElement *meHBEneMap_, *meHEEneMap_, *meHOEneMap_, *meHFEneMap_;
  MonitorElement *meHBEneSum_, *meHEEneSum_, *meHOEneSum_, *meHFEneSum_;
  MonitorElement *meHBEneSum_vs_ieta_, *meHEEneSum_vs_ieta_, *meHOEneSum_vs_ieta_, *meHFEneSum_vs_ieta_;
  MonitorElement *meHBTimHit_, *meHETimHit_, *meHOTimHit_, *meHFTimHit_;
  MonitorElement *meHBEneHit2_, *meHEEneHit2_, *meHOEneHit2_, *meHFEneHit2_;
  MonitorElement *meHBL10Ene_, *meHEL10Ene_, *meHOL10Ene_, *meHFL10Ene_;
  MonitorElement *meHBL10EneP_, *meHEL10EneP_, *meHOL10EneP_, *meHFL10EneP_;
  MonitorElement *meHEP17EneHit_, *meHEP17EneHit2_;
};

#endif
