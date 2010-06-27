#ifndef SimG4CMS_CaloSimHitStudy_H
#define SimG4CMS_CaloSimHitStudy_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <TH1F.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class CaloSimHitStudy: public edm::EDAnalyzer{
public:

  CaloSimHitStudy(const edm::ParameterSet& ps);
  ~CaloSimHitStudy() {}

protected:

  void beginJob () {}
  void endJob   () {}
  void analyze  (const edm::Event& e, const edm::EventSetup& c);

  void analyzeHits  (std::vector<PCaloHit> &, int);
  void analyzeHits  (edm::Handle<edm::PSimHitContainer>&, int);
  void analyzeHits  (std::vector<PCaloHit> &, std::vector<PCaloHit> &, std::vector<PCaloHit> &);

private:

  std::string    sourceLabel, g4Label, hitLab[4];
  std::string    muonLab[3], tkHighLab[6], tkLowLab[6];
  double         tmax_, eMIP_;

  TH1F           *hit_[9],  *time_[9], *edepEM_[9], *edepHad_[9], *edep_[9];
  TH1F           *etot_[9], *etotg_[9], *timeAll_[9], *hitMu, *hitHigh;
  TH1F           *hitLow, *eneInc_, *etaInc_, *phiInc_, *ptInc_;
  TH1F           *hitTk_[15], *edepTk_[15], *tofTk_[15], *edepC_[9],*edepT_[9];
};

#endif
