#ifndef GEMDigiValid_h
#define GEMDigiValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <iostream>
#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

class GEMDigiValid: public edm::EDAnalyzer
{

public:

  GEMDigiValid(const edm::ParameterSet& ps);
  ~GEMDigiValid();

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginJob();
  void endJob(void);
  void beginRun(edm::Run const&, edm::EventSetup const&);
  void endRun(edm::Run const&, edm::EventSetup const&);

private:
  MonitorElement* gemBxDist;
  MonitorElement* gemStripProf;
  MonitorElement* Res_gem;
  MonitorElement* xyview_simHits;
  MonitorElement* xyview_gemDigis;
//  MonitorElement* rzview_gem;
  MonitorElement* rzview_simHits;
  MonitorElement* rzview_gemDigis;
  MonitorElement* noiseGemCLS;
  MonitorElement* clsGEMs;
  MonitorElement* particleIDs;

//  int countEvent;

  DQMStore* dbe_;
  std::string outputFile_;
  std::string digiLabel;

  //particle types
  std::set<int> pidsSet;

  //Tokens for accessing run data. Used for passing to edm::Event. - stanislav
  edm::EDGetTokenT<edm::PSimHitContainer> simHitToken;
  edm::EDGetTokenT<GEMDigiCollection> gemDigiToken;
};

#endif
