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
  int FillHitValHist (int side,int section,int channel,double energy,double time);

private:
  double enetotEmN, enetotHadN,enetotN;  
  double enetotEmP,enetotHadP,enetotP;
  double enetot;


  std::string    g4Label, zdcHits, outFile_;
  bool           verbose_, checkHit_;
  DQMStore       *dbe_;

  MonitorElement *meAllZdcNHit_, *meBadZdcDetHit_, *meBadZdcSecHit_, *meBadZdcIdHit_;
  MonitorElement *meZdcNHit_,*meZdcDetectHit_,*meZdcSideHit_,*meZdcETime_;
  MonitorElement *meZdcNHitEM_,*meZdcNHitHad_,*meZdcNHitLum_,*meZdc10Ene_;
  MonitorElement *meZdcSectionHit_,*meZdcChannelHit_,*meZdcEnergyHit_,*meZdcTimeWHit_;
  MonitorElement *meZdcHadEnergyHit_, *meZdcEMEnergyHit_, *meZdcTimeHit_, *meZdcHadL10EneP_;
  MonitorElement *meZdc10EneP_, *meZdcEHadCh_, *meZdcEEMCh_,*meZdcEML10EneP_;
  MonitorElement *meZdcEneEmN1_,*meZdcEneEmN2_,*meZdcEneEmN3_,*meZdcEneEmN4_,*meZdcEneEmN5_; 
  MonitorElement *meZdcEneHadN1_,*meZdcEneHadN2_,*meZdcEneHadN3_,*meZdcEneHadN4_;
  MonitorElement *meZdcEneTEmN1_,*meZdcEneTEmN2_,*meZdcEneTEmN3_,*meZdcEneTEmN4_,*meZdcEneTEmN5_; 
  MonitorElement *meZdcEneTHadN1_,*meZdcEneTHadN2_,*meZdcEneTHadN3_,*meZdcEneTHadN4_;
  MonitorElement *meZdcEneHadNTot_,*meZdcEneEmNTot_,*meZdcEneNTot_;
  MonitorElement *meZdcCorEEmNEHadN_;
  MonitorElement *meZdcEneEmP1_,*meZdcEneEmP2_,*meZdcEneEmP3_,*meZdcEneEmP4_,*meZdcEneEmP5_; 
  MonitorElement *meZdcEneHadP1_,*meZdcEneHadP2_,*meZdcEneHadP3_,*meZdcEneHadP4_;
  MonitorElement *meZdcEneTEmP1_,*meZdcEneTEmP2_,*meZdcEneTEmP3_,*meZdcEneTEmP4_,*meZdcEneTEmP5_; 
  MonitorElement *meZdcEneTHadP1_,*meZdcEneTHadP2_,*meZdcEneTHadP3_,*meZdcEneTHadP4_;
  MonitorElement *meZdcEneHadPTot_,*meZdcEneEmPTot_, *meZdcEnePTot_;
  MonitorElement *meZdcCorEEmPEHadP_,*meZdcCorEtotNEtotP_,*meZdcEneTot_;
};

#endif
