#ifndef SimG4CMS_HcalSimHitStudy_H
#define SimG4CMS_HcalSimHitStudy_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class HcalSimHitStudy: public DQMEDAnalyzer {
public:

  HcalSimHitStudy(const edm::ParameterSet& ps);
  ~HcalSimHitStudy();

  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const & , edm::EventSetup const & );

protected:

  //void endJob   ();
  void analyze  (const edm::Event& e, const edm::EventSetup& c);

  void analyzeHits  (std::vector<PCaloHit> &);

private:

  std::string    g4Label, hcalHits, outFile_;
  bool           verbose_, checkHit_;

  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;

  MonitorElement *meAllNHit_, *meBadDetHit_, *meBadSubHit_, *meBadIdHit_;
  MonitorElement *meHBNHit_, *meHENHit_, *meHONHit_, *meHFNHit_;
  MonitorElement *meDetectHit_, *meSubdetHit_, *meDepthHit_, *meEtaHit_;
  MonitorElement *mePhiHit_, *mePhiHitb_, *meEnergyHit_, *meTimeHit_, *meTimeWHit_;
  MonitorElement *meHBDepHit_, *meHEDepHit_, *meHODepHit_, *meHFDepHit_;
  MonitorElement *meHBEtaHit_, *meHEEtaHit_, *meHOEtaHit_, *meHFEtaHit_;
  MonitorElement *meHBPhiHit_, *meHEPhiHit_, *meHOPhiHit_, *meHFPhiHit_;
  MonitorElement *meHBEneHit_, *meHEEneHit_, *meHOEneHit_, *meHFEneHit_;
  MonitorElement *meHBTimHit_, *meHETimHit_, *meHOTimHit_, *meHFTimHit_;
  MonitorElement *meHBEneHit2_, *meHEEneHit2_, *meHOEneHit2_, *meHFEneHit2_;
  MonitorElement *meHBL10Ene_, *meHEL10Ene_, *meHOL10Ene_, *meHFL10Ene_;
  MonitorElement *meHBL10EneP_, *meHEL10EneP_, *meHOL10EneP_, *meHFL10EneP_;

};

#endif
