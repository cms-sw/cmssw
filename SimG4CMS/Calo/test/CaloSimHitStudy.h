#ifndef SimG4CMS_CaloSimHitStudy_H
#define SimG4CMS_CaloSimHitStudy_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

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

  void beginJob (const edm::EventSetup& c) {}
  void endJob   () {}
  void analyze  (const edm::Event& e, const edm::EventSetup& c);

  void analyzeHits  (std::vector<PCaloHit> &, int);

private:

  std::string    sourceLabel, g4Label, hitLab[4];
  std::string    muonLab[3], tkHighLab[6], tkLowLab[6];

  TH1F           *hit_[7],  *time_[7], *edepEM_[7], *edepHad_[7], *edep_[7];
  TH1F           *etot_[7], *etotg_[7], *timeAll_[7], *hitMu, *hitHigh;
  TH1F           *hitLow, *eneInc_, *etaInc_, *phiInc_, *ptInc_;
};

#endif
