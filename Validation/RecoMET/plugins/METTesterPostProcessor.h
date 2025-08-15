#ifndef METTESTERPOSTPROCESSOR_H
#define METTESTERPOSTPROCESSOR_H
// author: Matthias Weber, Feb 2015

// user include files
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Validation/RecoMET/plugins/METTester.h"

//
// class declaration
//
class METTesterPostProcessor : public DQMEDHarvester {
public:
  explicit METTesterPostProcessor(const edm::ParameterSet &);
  ~METTesterPostProcessor() override;

private:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  std::vector<std::string> met_dirs;

  void mFillAggrHistograms(std::string, DQMStore::IGetter&);

  template <int S>
  using ElemArr = std::array<MonitorElement*, S>;
	
  static constexpr int mNMETBins = METTester::mNMETBins;
  static constexpr auto mMETBins = METTester::mMETBins;
  ElemArr<mNMETBins> mMET_METBins;
  ElemArr<mNMETBins> mMETDiff_GenMETTrue_METBins;
  ElemArr<mNMETBins> mMETRatio_GenMETTrue_METBins;
  ElemArr<mNMETBins> mMETDeltaPhi_GenMETTrue_METBins;

  static constexpr int mNEtaBins = METTester::mNEtaBins;
  static constexpr auto mEtaBins = METTester::mEtaBins;
  ElemArr<mNEtaBins> mMET_EtaBins;
  ElemArr<mNEtaBins> mMETDiff_GenMETTrue_EtaBins;
  ElemArr<mNEtaBins> mMETRatio_GenMETTrue_EtaBins;
  ElemArr<mNEtaBins> mMETDeltaPhi_GenMETTrue_EtaBins;
  
  static constexpr int mNPhiBins = METTester::mNPhiBins;
  static constexpr auto mPhiBins = METTester::mPhiBins;
  ElemArr<mNPhiBins> mMET_PhiBins;
  ElemArr<mNPhiBins> mMETDiff_GenMETTrue_PhiBins;
  ElemArr<mNPhiBins> mMETRatio_GenMETTrue_PhiBins;
  ElemArr<mNPhiBins> mMETDeltaPhi_GenMETTrue_PhiBins;
  
  MonitorElement *mMETDiffAggr_METBins, *mMETDiffAggr_EtaBins, *mMETDiffAggr_PhiBins;
  MonitorElement *mMETRespAggr_METBins, *mMETRespAggr_EtaBins, *mMETRespAggr_PhiBins;
  MonitorElement *mMETResolAggr_METBins, *mMETResolAggr_EtaBins, *mMETResolAggr_PhiBins;
  MonitorElement *mMETSignAggr_METBins, *mMETSignAggr_EtaBins, *mMETSignAggr_PhiBins;
};

#endif
