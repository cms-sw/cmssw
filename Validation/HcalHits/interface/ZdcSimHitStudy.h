#ifndef SimG4CMS_ZdcSimHitStudy_H
#define SimG4CMS_ZdcSimHitStudy_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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

class ZdcSimHitStudy: public edm::EDAnalyzer{
public:

  ZdcSimHitStudy(const edm::ParameterSet& ps);
  ~ZdcSimHitStudy();

protected:

  void beginJob ();
  void endJob   ();
  void analyze  (const edm::Event& e, const edm::EventSetup& c);

  void analyzeHits  (std::vector<PCaloHit> &);

private:

  std::string    g4Label, zdcHits, outFile_;
  bool           verbose_, checkHit_;
  DQMStore       *dbe_;

  MonitorElement *meAllZdcNHit_, *meBadZdcDetHit_, *meBadZdcSecHit_, *meBadZdcIdHit_;
  MonitorElement *meZdcNHit_,*meZdcDetectHit_, *meZdcSideHit_,*meZdcETime_;
  MonitorElement *meZdcNHitEM_,*meZdcNHitHad_,*meZdcNHitLum_,*meZdc10Ene_;
  MonitorElement *meZdcSectionHit_,*meZdcChannelHit_,*meZdcEnergyHit_,*meZdcTimeWHit_;
  MonitorElement *meZdcHadEnergyHit_, *meZdcEMEnergyHit_, *meZdcTimeHit_, *meZdcHadL10EneP_;
  MonitorElement *meZdc10EneP_, *meZdcEHadCh_, *meZdcEEMCh_,*meZdcEML10EneP_;

};

#endif
