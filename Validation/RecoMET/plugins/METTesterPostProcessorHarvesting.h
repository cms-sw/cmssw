#ifndef METTESTERPOSTPROCESSORHARVESTING_H
#define METTESTERPOSTPROCESSORHARVESTING_H

// author: Matthias Weber, Feb 2015

// system include files
#include <cmath>
#include <cstdio>
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//
// class decleration
//

class METTesterPostProcessorHarvesting : public DQMEDHarvester {
public:
  explicit METTesterPostProcessorHarvesting(const edm::ParameterSet &);
  ~METTesterPostProcessorHarvesting() override;

private:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  edm::InputTag inputMETLabelRECO_;
  edm::InputTag inputMETLabelMiniAOD_;

  std::vector<std::string> met_dirs;

  MonitorElement *mMET_MiniAOD_over_Reco;
  MonitorElement *mMETPhi_MiniAOD_over_Reco;
  MonitorElement *mSumET_MiniAOD_over_Reco;
  MonitorElement *mPFPhotonEtFraction_MiniAOD_over_Reco;
  MonitorElement *mPFNeutralHadronEtFraction_MiniAOD_over_Reco;
  MonitorElement *mPFChargedHadronEtFraction_MiniAOD_over_Reco;
  MonitorElement *mPFHFHadronEtFraction_MiniAOD_over_Reco;
  MonitorElement *mPFHFEMEtFraction_MiniAOD_over_Reco;
  MonitorElement *mMETDifference_GenMETTrue_MiniAOD_over_Reco;
  MonitorElement *mMETDeltaPhi_GenMETTrue_MiniAOD_over_Reco;
  MonitorElement *mMETDifference_GenMETTrue_MET20to40_MiniAOD_over_Reco;
  MonitorElement *mMETDifference_GenMETTrue_MET100to150_MiniAOD_over_Reco;
  MonitorElement *mMETDifference_GenMETTrue_MET300to400_MiniAOD_over_Reco;
};

#endif
