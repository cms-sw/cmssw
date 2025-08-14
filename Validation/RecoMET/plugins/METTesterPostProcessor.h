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

  void FillMETRes(std::string metdir, DQMStore::IGetter &);

  static constexpr int mNMETBins = METTester::mNMETBins;
  static constexpr auto mMETBins = METTester::mMETBins;
  std::array<MonitorElement *, mNMETBins> mMETDifference_GenMETTrue_METBins;

  static constexpr int mNEtaBins = METTester::mNEtaBins;
  static constexpr auto mEtaBins = METTester::mEtaBins;
  std::array<MonitorElement *, mNEtaBins> mMETDifference_GenMETTrue_EtaBins;
  
  static constexpr int mNPhiBins = METTester::mNPhiBins;
  static constexpr auto mPhiBins = METTester::mPhiBins;
  std::array<MonitorElement *, mNPhiBins> mMETDifference_GenMETTrue_PhiBins;
  
  MonitorElement *mMETDiffAggr_METBins;
  MonitorElement *mMETDiffAggr_EtaBins;
  MonitorElement *mMETDiffAggr_PhiBins;
};

#endif
