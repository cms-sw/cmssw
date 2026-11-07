#ifndef METTESTERPOSTPROCESSOR_H
#define METTESTERPOSTPROCESSOR_H

// user include files
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Validation/RecoMET/plugins/METTester.h"

#include <optional>

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

  using MElem = MonitorElement;

  static constexpr int mNMETBins = 11;
  const std::vector<float> mMETBins = {{0., 20., 40., 60., 80., 100., 150., 200., 300., 400., 500., 1000.}};

  static constexpr int mNPhiBins = 6;
  const std::vector<float> mPhiBins = {{-3.15, -2., -1., 0., 1., 2., 3.15}};

  // 2D histograms filled by METTester and required by the post-processing
  struct Inputs {
    MElem* genMETPhi_vs_MET;
    MElem* genMETTrue_vs_GenMETPhi;
    MElem* metDiff_vs_GenMETTrue;
    MElem* metDiff_vs_GenMETPhi;
    MElem* metRatio_vs_GenMETTrue;
    MElem* metRatio_vs_GenMETPhi;
    MElem* metDeltaPhi_vs_GenMETTrue;
    MElem* metDeltaPhi_vs_GenMETPhi;
  };

  // statistics of the 1D projection of a 2D histogram in a slice
  struct ProjStats {
    float mean, meanErr, rms, rmsErr;
  };

  std::string runDir;

  float mEpsilonFloat = std::numeric_limits<float>::epsilon();

  // methods
  std::optional<Inputs> mGetInputs(const std::string& metdir, DQMStore::IGetter&);
  void mFillAggrHistograms(const std::string& metdir, const Inputs&, DQMStore::IBooker&);
  float mComputeSignErr(float significance, float metRMS, float metMean, float metRMSError);
  // `axis` is the axis along which the slices defined by `bins` are taken; the other axis is projected
  std::vector<ProjStats> projectionMeanAndRMS(MElem* src, const std::vector<float>& bins, const std::string& axis);
  void fillProjectionHisto(MElem* src, MElem* dest, const std::vector<float>& bins);
};

#endif  // METTESTERPOSTPROCESSOR_H
