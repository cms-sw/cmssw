#ifndef JETTESTERPOSTPROCESSOR_H
#define JETTESTERPOSTPROCESSOR_H

// author: Matthias Weber, Feb 2015

// system include files
#include <memory>
#include <stdio.h>
#include <math.h>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
//
// class decleration
//

class JetTesterPostProcessor : public DQMEDHarvester {
   public:
      explicit JetTesterPostProcessor(const edm::ParameterSet&);
      ~JetTesterPostProcessor();

   private:
      virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) ;

      edm::InputTag inputJetLabelRECO_;
      edm::InputTag inputJetLabelMiniAOD_;

      std::vector<std::string> jet_dirs;

      MonitorElement* mGenPt_MiniAOD_over_Reco;
      MonitorElement* mGenPhi_MiniAOD_over_Reco;
      MonitorElement* mGenEta_MiniAOD_over_Reco;
      MonitorElement* mPt_MiniAOD_over_Reco;
      MonitorElement* mPhi_MiniAOD_over_Reco;
      MonitorElement* mEta_MiniAOD_over_Reco;
      MonitorElement* mCorrJetPt_MiniAOD_over_Reco;
      MonitorElement* mCorrJetPhi_MiniAOD_over_Reco;
      MonitorElement* mCorrJetEta_MiniAOD_over_Reco;
      MonitorElement* mPtCorrOverReco_Eta_20_40_MiniAOD_over_Reco;
      MonitorElement* mPtCorrOverReco_Eta_200_600_MiniAOD_over_Reco;
      MonitorElement* mPtCorrOverReco_Eta_1500_3500_MiniAOD_over_Reco;
      MonitorElement* mPtCorrOverGen_GenEta_40_200_MiniAOD_over_Reco;
      MonitorElement* mPtCorrOverGen_GenEta_600_1500_MiniAOD_over_Reco;
      MonitorElement* mDeltaEta_MiniAOD_over_Reco;
      MonitorElement* mDeltaPhi_MiniAOD_over_Reco;
      MonitorElement* mDeltaPt_MiniAOD_over_Reco;
      MonitorElement* mMjj_MiniAOD_over_Reco;
      MonitorElement* mNJets40_MiniAOD_over_Reco;
      MonitorElement* mchargedHadronMultiplicity_MiniAOD_over_Reco;
      MonitorElement* mneutralHadronMultiplicity_MiniAOD_over_Reco;
      MonitorElement* mphotonMultiplicity_MiniAOD_over_Reco;
      MonitorElement* mphotonEnergyFraction_MiniAOD_over_Reco;
      MonitorElement* mneutralHadronEnergyFraction_MiniAOD_over_Reco;
      MonitorElement* mchargedHadronEnergyFraction_MiniAOD_over_Reco;

};

#endif
