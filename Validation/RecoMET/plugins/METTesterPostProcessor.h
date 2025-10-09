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
  explicit METTesterPostProcessor(const edm::ParameterSet&);
  ~METTesterPostProcessor() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  std::vector<std::string> met_dirs;

  using MElem = MonitorElement;

  void mFillAggrHistograms(std::string, DQMStore::IGetter&);
  bool mCheckHisto(MElem* h);

  template <typename T>
  using ArrayVariant = std::variant<std::array<T, METTester::mNMETBins + 1>, std::array<T, METTester::mNPhiBins + 1>>;

  // reading
  template <typename T>
  T mArrayIdx(const ArrayVariant<T>& arr, unsigned idx) {
    return std::visit([idx](const auto& x) { return x.at(idx); }, arr);
  }
  // assigning
  template <typename T>
  auto& mArrayIdx(ArrayVariant<T>& arr, unsigned idx) {
    return std::visit([idx](auto& x) -> auto& { return x.at(idx); }, arr);
  }

  std::unordered_map<std::string, unsigned> mNBins = {{"MET", METTester::mNMETBins}, {"Phi", METTester::mNPhiBins}};
  std::unordered_map<std::string, ArrayVariant<float>> mEdges = {
      {"MET", std::array<float, METTester::mNMETBins + 1>{METTester::mMETBins}},
      {"Phi", std::array<float, METTester::mNPhiBins + 1>{METTester::mPhiBins}}};

  using ElemMap = std::unordered_map<std::string, MElem*>;  // one entry per bin type, for instance "MET" and "Phi"
  using ElemMapArr =
      std::unordered_map<std::string, ArrayVariant<MElem*>>;  // one entry per bin type, for instance "MET" and "Phi"

  ElemMapArr mMET;
  ElemMapArr mMETDiff_GenMETTrue;
  ElemMapArr mMETRatio_GenMETTrue;
  ElemMapArr mMETDeltaPhi_GenMETTrue;

  ElemMap mMETDiffAggr;
  ElemMap mMETRespAggr;
  ElemMap mMETResolAggr;
  ElemMap mMETSignAggr;

  std::string runDir;

  float mEpsilonFloat = std::numeric_limits<float>::epsilon();
  double mEpsilonDouble = std::numeric_limits<double>::epsilon();
};

#endif
