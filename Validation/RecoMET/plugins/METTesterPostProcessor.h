#ifndef METTESTERPOSTPROCESSOR_H
#define METTESTERPOSTPROCESSOR_H

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
  explicit METTesterPostProcessor(const edm::ParameterSet&);
  ~METTesterPostProcessor() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  std::vector<std::string> met_dirs;

  using MElem = MonitorElement;
	
  static constexpr int mNMETBins = 11;
  const std::vector<float> mMETBins = {{0., 20., 40., 60., 80., 100., 150., 200., 300., 400., 500., 1000.}};

  static constexpr int mNPhiBins = 6;
  const std::vector<float> mPhiBins = {{-3.15, -2., -1., 0., 1., 2., 3.15}};

  using ElemMap = std::unordered_map<std::string, MElem*>;  // one entry per bin type, for instance "MET" and "Phi"

  MElem *mGenMETTrue_vs_MET;
  MElem *mGenMETPhi_vs_MET;
  MElem *mGenMETTrue_vs_mGenMETPhi;
  MElem *mMETDiff_vs_GenMETTrue;
  MElem *mMETDiff_vs_GenMETPhi;
  MElem *mMETRatio_vs_GenMETTrue;
  MElem *mMETRatio_vs_GenMETPhi;
  MElem *mMETDeltaPhi_vs_GenMETTrue;
  MElem *mMETDeltaPhi_vs_GenMETPhi;

  ElemMap mMETDiffAggr;
  ElemMap mMETDeltaPhiAggr;
  ElemMap mMETRespAggr;
  ElemMap mMETResolAggr, mMETGenResolAggr, mMETResolDiffAggr;
  ElemMap mMETSignAggr, mMETGenSignAggr, mMETSignDiffAggr;

  std::string runDir;

  float mEpsilonFloat = std::numeric_limits<float>::epsilon();
  double mEpsilonDouble = std::numeric_limits<double>::epsilon();

  // methods
  void mFillAggrHistograms(std::string, DQMStore::IGetter&);
  bool mCheckHisto(MElem* h);
  float mComputeSignErr(float significance, float metRMS, float metMean, float metRMSError);
  std::vector<std::unordered_map<std::string,float>> projectionMeanAndRMS(MElem* src, const std::vector<float>& bins, std::string axis = "X");
  void fillProjectionHisto(MElem* src, MElem* dest, const std::vector<float>& bins);
  void logDebugEmptyHisto(std::string);
};

#endif
