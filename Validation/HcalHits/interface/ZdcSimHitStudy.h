////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Package:    ZdcSimHitStudy
// Class:      ZdcSimHitStudy
//
/*
 Description:
              This code has been developed to be a check for the ZDC sim. In
 2009, it was found that the ZDC Simulation was unrealistic and needed repair.
 The aim of this code is to show the user the input and output of a ZDC MinBias
 simulation.

 Implementation:
      First a MinBias simulation should be run, it could be pythia,hijin,or
 hydjet. This will output a .root file which should have information about
 recoGenParticles, hcalunsuppresseddigis, and g4SimHits_ZDCHits. Use this .root
 file as the input into the cfg.py which is found in the main directory of this
 package. This output will be another .root file which is meant to be viewed in
 a TBrowser.

*/
//
// Original Author: Jaime Gomez (U. of Maryland) with SIGNIFICANT assistance of
// Dr. Jefferey Temple (U. of Maryland) Adapted from: E. Garcia-Solis' (CSU)
// original code
//
//         Created:  Summer 2012
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SimG4CMS_ZdcSimHitStudy_H
#define SimG4CMS_ZdcSimHitStudy_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ZdcSimHitStudy : public DQMEDAnalyzer {
public:
  ZdcSimHitStudy(const edm::ParameterSet &ps);
  ~ZdcSimHitStudy() override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  void analyzeHits(std::vector<PCaloHit> &);
  int FillHitValHist(int side, int section, int channel, double energy, double time);

private:
  double enetotEmN, enetotHadN, enetotN;
  double enetotEmP, enetotHadP, enetotP;
  double enetot;

  /////////////////////////////////////////
  //#   Below all the monitoring elements #
  //# are simply the plots "code names"  #
  //# they will be filled in the .cc file #
  /////////////////////////////////////////

  std::string g4Label, zdcHits, outFile_;
  edm::EDGetTokenT<reco::GenParticleCollection> tok_gen_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;
  bool verbose_, checkHit_;

  MonitorElement *meAllZdcNHit_, *meBadZdcDetHit_, *meBadZdcSecHit_, *meBadZdcIdHit_;
  MonitorElement *meZdcNHit_, *meZdcDetectHit_, *meZdcSideHit_, *meZdcETime_;
  MonitorElement *meZdcNHitEM_, *meZdcNHitHad_, *meZdcNHitLum_, *meZdc10Ene_;
  MonitorElement *meZdcSectionHit_, *meZdcChannelHit_, *meZdcEnergyHit_, *meZdcTimeWHit_;
  MonitorElement *meZdcHadEnergyHit_, *meZdcEMEnergyHit_, *meZdcTimeHit_, *meZdcHadL10EneP_;
  MonitorElement *meZdc10EneP_, *meZdcEHadCh_, *meZdcEEMCh_, *meZdcEML10EneP_;
  MonitorElement *meZdcEneEmN1_, *meZdcEneEmN2_, *meZdcEneEmN3_, *meZdcEneEmN4_, *meZdcEneEmN5_;
  MonitorElement *meZdcEneHadN1_, *meZdcEneHadN2_, *meZdcEneHadN3_, *meZdcEneHadN4_;
  MonitorElement *meZdcEneTEmN1_, *meZdcEneTEmN2_, *meZdcEneTEmN3_, *meZdcEneTEmN4_, *meZdcEneTEmN5_;
  MonitorElement *meZdcEneTHadN1_, *meZdcEneTHadN2_, *meZdcEneTHadN3_, *meZdcEneTHadN4_;
  MonitorElement *meZdcEneHadNTot_, *meZdcEneEmNTot_, *meZdcEneNTot_;
  MonitorElement *meZdcCorEEmNEHadN_;
  MonitorElement *meZdcEneEmP1_, *meZdcEneEmP2_, *meZdcEneEmP3_, *meZdcEneEmP4_, *meZdcEneEmP5_;
  MonitorElement *meZdcEneHadP1_, *meZdcEneHadP2_, *meZdcEneHadP3_, *meZdcEneHadP4_;
  MonitorElement *meZdcEneTEmP1_, *meZdcEneTEmP2_, *meZdcEneTEmP3_, *meZdcEneTEmP4_, *meZdcEneTEmP5_;
  MonitorElement *meZdcEneTHadP1_, *meZdcEneTHadP2_, *meZdcEneTHadP3_, *meZdcEneTHadP4_;
  MonitorElement *meZdcEneHadPTot_, *meZdcEneEmPTot_, *meZdcEnePTot_;
  MonitorElement *meZdcCorEEmPEHadP_, *meZdcCorEtotNEtotP_, *meZdcEneTot_;

  ///////////////////New Plots/////////////

  ////////GenParticle Plots///////
  MonitorElement *genpart_Pi0F;
  MonitorElement *genpart_Pi0F_energydist;
  MonitorElement *genpart_Pi0B;
  MonitorElement *genpart_Pi0B_energydist;
  MonitorElement *genpart_NeutF;
  MonitorElement *genpart_NeutF_energydist;
  MonitorElement *genpart_NeutB;
  MonitorElement *genpart_NeutB_energydist;
  MonitorElement *genpart_GammaF;
  MonitorElement *genpart_GammaF_energydist;
  MonitorElement *genpart_GammaB;
  MonitorElement *genpart_GammaB_energydist;
  //////////////////////////////

  // N counts plots
  MonitorElement *genpart_Pi0F_counts;
  MonitorElement *genpart_Pi0B_counts;
  MonitorElement *genpart_NeutF_counts;
  MonitorElement *genpart_NeutB_counts;
  MonitorElement *genpart_GammaF_counts;
  MonitorElement *genpart_GammaB_counts;
};

#endif
