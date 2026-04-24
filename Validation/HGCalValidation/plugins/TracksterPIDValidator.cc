/**
DQM plots for trackster PID, based on simulation truth
Takes as input a trackster collection and a mask to build the baseline (efficiency denominator)
 Computes the efficiency as EFF=#(baseline & passing PID cut) / #(baseline)
Another mask is used to define a "fake" baseline region (populated by unmatched tracksters),
 the fake rate is then defined as FR=#(fake baseline & passing signal PID cut)/#(fake baseline)

Author: Théo Cuisset (LLR)
*/
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"

using namespace ticl;

struct Histograms_TracksterPIDValidation {
  dqm::reco::MonitorElement* pt_eta_pid_;  // 3D histo pt-abs(eta)-PID value

  dqm::reco::MonitorElement* pt_eta_reco2SimSelected_;  // 2D pt-eta after selection on reco2Sim score but before PID cut
  dqm::reco::MonitorElement* pt_eta_noReco2SimSelection_;  // 2D pt-eta before selection on reco2Sim score (->denominator)
  dqm::reco::MonitorElement* pt_eta_pidNum_;               // 2D pt-eta after PID cut

  dqm::reco::MonitorElement* pt_eta_pid_fakes_;     // 3D histo pt-abs(eta)-PID value in fakes region
  dqm::reco::MonitorElement* pt_eta_fakes_;         // 2D pt-eta fakes selection before PID cut
  dqm::reco::MonitorElement* pt_eta_fakes_pid_Num;  // 2D pt-eta fakes selection after PID cut
};

class TICLTracksterPIDValidation : public DQMGlobalEDAnalyzer<Histograms_TracksterPIDValidation> {
public:
  explicit TICLTracksterPIDValidation(const edm::ParameterSet&);
  ~TICLTracksterPIDValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&,
                      edm::Run const&,
                      edm::EventSetup const&,
                      Histograms_TracksterPIDValidation&) const override;

  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms_TracksterPIDValidation const&) const override;

  const std::string folder_;
  edm::EDGetTokenT<ticl::TracksterCollection> tracksters_token_;
  edm::EDGetTokenT<std::vector<int>> tracksters_mask_token_;
  edm::EDGetTokenT<std::vector<int>> tracksters_mask_fakes_token_;

  const bool doFakes_;

  const double pidCut_;

  const std::vector<ticl::Trackster::ParticleType> pidsToConsider_;
  const std::vector<double> pidBins_;
};

TICLTracksterPIDValidation::TICLTracksterPIDValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      tracksters_token_(consumes<ticl::TracksterCollection>(iConfig.getParameter<edm::InputTag>("tracksters"))),
      doFakes_(iConfig.exists("tracksterMaskFakes")),
      pidCut_(iConfig.getParameter<double>("pidCut")),
      pidsToConsider_({ticl::Trackster::ParticleType::electron, ticl::Trackster::ParticleType::photon}),
      pidBins_(iConfig.getParameter<std::vector<double>>("pidBins")) {
  if (iConfig.exists("tracksterMask"))
    tracksters_mask_token_ = consumes<std::vector<int>>(iConfig.getParameter<edm::InputTag>("tracksterMask"));
  if (iConfig.exists("tracksterMaskFakes"))
    tracksters_mask_fakes_token_ =
        consumes<std::vector<int>>(iConfig.getParameter<edm::InputTag>("tracksterMaskFakes"));
  else
    edm::LogInfo("PIDValidation") << "Not using any mask for fakes, will not run validation for fakes";
}

void TICLTracksterPIDValidation::dqmAnalyze(edm::Event const& iEvent,
                                            edm::EventSetup const& iSetup,
                                            Histograms_TracksterPIDValidation const& histos) const {
  ticl::TracksterCollection const& tracksters = iEvent.get(tracksters_token_);
  std::vector<int> const& tracksterMask = tracksters_mask_token_.isUninitialized()
                                              ? std::vector<int>(tracksters.size(), 1)
                                              : iEvent.get(tracksters_mask_token_);
  if (tracksters_mask_token_.isUninitialized())
    edm::LogWarning("PIDValidation") << "Not using any mask";

  assert(tracksterMask.size() == tracksters.size());

  auto doFill = [](dqm::reco::MonitorElement* h, Trackster const& ts) {
    h->Fill(ts.raw_pt(), std::abs(ts.barycenter().eta()));
  };

  // Signal
  for (std::size_t i = 0; i < tracksters.size(); i++) {
    if (tracksterMask[i] == 1)
      continue;  // Trackster is not the best-matched one to CaloParticle
    ticl::Trackster const& ts = tracksters[i];

    doFill(histos.pt_eta_noReco2SimSelection_, ts);

    if (tracksterMask[i] != 0)
      continue;  // Trackster is not the best-matched one to CaloParticle

    doFill(histos.pt_eta_reco2SimSelected_, ts);

    const double pidValue = std::transform_reduce(
        pidsToConsider_.begin(), pidsToConsider_.end(), 0., std::plus<>{}, [&ts](Trackster::ParticleType partType) {
          return ts.id_probability(partType);
        });

    histos.pt_eta_pid_->Fill(ts.raw_pt(), std::abs(ts.barycenter().eta()), pidValue);

    if (pidValue > pidCut_)
      doFill(histos.pt_eta_pidNum_, ts);
  }

  // Fakes
  if (!doFakes_) {
    return;
  }

  std::vector<int> const& tracksterMaskFakes = iEvent.get(tracksters_mask_fakes_token_);
  assert(tracksterMaskFakes.size() == tracksters.size());

  for (std::size_t i = 0; i < tracksters.size(); i++) {
    if (tracksterMaskFakes[i] != 0)
      continue;  // Trackster is not in the fakes mask (too signal like for example)
    ticl::Trackster const& ts = tracksters[i];

    doFill(histos.pt_eta_fakes_, ts);

    const double pidValue = std::transform_reduce(
        pidsToConsider_.begin(), pidsToConsider_.end(), 0., std::plus<>{}, [&ts](Trackster::ParticleType partType) {
          return ts.id_probability(partType);
        });
    histos.pt_eta_pid_fakes_->Fill(ts.raw_pt(), std::abs(ts.barycenter().eta()), pidValue);

    if (pidValue > pidCut_)
      doFill(histos.pt_eta_fakes_pid_Num, ts);
  }
}

void TICLTracksterPIDValidation::bookHistograms(DQMStore::IBooker& ibook,
                                                edm::Run const& run,
                                                edm::EventSetup const& iSetup,
                                                Histograms_TracksterPIDValidation& histos) const {
  ibook.setCurrentFolder(folder_);
  // ROOT histograms have bin edges in double, but CMSSW DQM wants float. But for PID we need double edges, so make two arrays
  constexpr std::array<double, 20> ptBins = {0., 0.5, 1.,  1.5, 2.,  2.5, 3.,  4.,  5.,  6.,
                                             7., 8.,  10., 12., 15., 20., 30., 40., 50., 100.};
  std::array<float, ptBins.size()> ptBinsF;
  std::copy(ptBins.begin(), ptBins.end(), ptBinsF.begin());  // convert to float

  //    constexpr std::array etaBins = {1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1};
  constexpr std::array<double, 9> etaBins = {1.6, 1.8, 2.1, 2.3, 2.5, 2.7, 2.8, 2.9, 3.};
  std::array<float, etaBins.size()> etaBinsF;
  std::copy(etaBins.begin(), etaBins.end(), etaBinsF.begin());  // convert to float

  auto make3D = [&](const char* name, const char* title) -> TH3F* {
    return new TH3F(name,
                    title,
                    ptBins.size() - 1,
                    ptBins.data(),  // pt
                    etaBins.size() - 1,
                    etaBins.data(),  //abs(eta)
                    pidBins_.size() - 1,
                    pidBins_.data()  //PID
    );
  };
  histos.pt_eta_pid_ = ibook.book3D("pt_eta_pid", make3D("pt_eta_pid", "Pt-abs(eta)-PID value for trackster (signal)"));
  histos.pt_eta_pid_fakes_ =
      ibook.book3D("pt_eta_pid_fakes", make3D("pt_eta_pid_fakes", "Pt-abs(eta)-PID value for trackster (fakes)"));

  histos.pt_eta_noReco2SimSelection_ =
      ibook.book2D("pt_eta_noReco2SimSelection",
                   "Pt-abs(eta) for the trackster best-associated (shared energy) to sim",
                   ptBinsF.size() - 1,
                   ptBinsF.data(),  // pt
                   etaBinsF.size() - 1,
                   etaBinsF.data()  //abs(eta)
      );
  histos.pt_eta_reco2SimSelected_ = ibook.book2D(
      "pt_eta_reco2SimSelected",
      "Pt-abs(eta) for the trackster best-associated (shared energy) to sim (additionally passing reco2sim cut)",
      ptBinsF.size() - 1,
      ptBinsF.data(),  // pt
      etaBinsF.size() - 1,
      etaBinsF.data()  //abs(eta)
  );

  histos.pt_eta_pidNum_ = ibook.book2D("pt_eta_pidNum",
                                       "Pt-abs(eta) for trackster after PID cut",
                                       ptBinsF.size() - 1,
                                       ptBinsF.data(),  // pt
                                       etaBinsF.size() - 1,
                                       etaBinsF.data()  //abs(eta)
  );

  // fakes
  if (doFakes_) {
    histos.pt_eta_fakes_ = ibook.book2D("pt_eta_fakes",
                                        "Pt-abs(eta) for the 'fake' tracksters before PID selection",
                                        ptBinsF.size() - 1,
                                        ptBinsF.data(),  // pt
                                        etaBinsF.size() - 1,
                                        etaBinsF.data()  //abs(eta)
    );
    histos.pt_eta_fakes_pid_Num =
        ibook.book2D("pt_eta_fakes_pid_Num",
                     "Pt-abs(eta) for the 'fake' tracksters after (signal) PID selection (numerator for fake rate)",
                     ptBinsF.size() - 1,
                     ptBinsF.data(),  // pt
                     etaBinsF.size() - 1,
                     etaBinsF.data()  //abs(eta)
        );
  }
}

void TICLTracksterPIDValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "HGCAL/TICLTracksterPIDValidation/");  // Please keep the trailing '/'
  desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTrackstersCLUE3DHigh"));
  desc.addOptional<edm::InputTag>("tracksterMask");
  desc.addOptional<edm::InputTag>("tracksterMaskFakes");
  desc.add<double>("pidCut", 0.5)->setComment("Cut on the PID score to apply while making PID efficiency plots");

  // The default was generated with a logit :  [1/(1+math.exp(-x)) for x in np.linspace(-10, 10, 99)]
  // clang-format off
  desc.add<std::vector<double>>("pidBins",
                                {0., 
                                  2.0611536181902037e-09, 3.100083662879109e-09, 4.662689198418738e-09, 7.012930265158089e-09, 1.054781666819802e-08, 1.58644720624746e-08, 2.3861001860734702e-08, 3.588820385960723e-08, 5.3977748963680536e-08, 8.118537690037458e-08, 1.2210708039159532e-07, 1.836554727399914e-07, 2.762274842111852e-07, 4.1546063580332945e-07, 6.248745604778494e-07, 9.398439742943763e-07, 1.413574198167425e-06, 2.1260885755925174e-06, 3.197745837255419e-06, 4.8095705106148075e-06, 7.233829979130707e-06, 1.0880021845800016e-05, 1.6364036499311468e-05, 2.4612170282493442e-05, 3.70175419513405e-05, 5.5675295113294124e-05, 8.373622806675614e-05, 0.00012593838443361062, 0.00018940594382518605, 0.00028484932661867926, 0.00042836689448632893, 0.000644147300607427, 0.000968517024997084, 0.0014559898567672233, 0.0021882795075905265, 0.0032876613859340125, 0.0049366352189008315, 0.007406529685701493, 0.011098377611465102, 0.016599683415824223, 0.02475963197867262, 0.03678076158088027, 0.0543132661326406, 0.07951320168980043, 0.11498363458710696, 0.16346724250610595, 0.22714729517594293, 0.30654399138843746, 0.39935261733925703, 0.5, 0.6006473826607429, 0.6934560086115625, 0.7728527048240571, 0.8365327574938946, 0.8850163654128934, 0.9204867983101995, 0.9456867338673594, 0.9632192384191197, 0.9752403680213275, 0.9834003165841759, 0.9889016223885349, 0.9925934703142986, 0.9950633647810992, 0.9967123386140659, 0.9978117204924094, 0.9985440101432327, 0.9990314829750029, 0.9993558526993926, 0.9995716331055137, 0.9997151506733813, 0.9998105940561749, 0.9998740616155662, 0.9999162637719331, 0.9999443247048868, 0.9999629824580487, 0.9999753878297175, 0.9999836359635007, 0.9999891199781543, 0.9999927661700209, 0.9999951904294894, 0.9999968022541628, 0.9999978739114245, 0.9999985864258019, 0.9999990601560258, 0.9999993751254396, 0.9999995845393642, 0.9999997237725157, 0.9999998163445274, 0.9999998778929196, 0.999999918814623, 0.9999999460222511, 0.9999999641117963, 0.999999976138998, 0.9999999841355278, 0.9999999894521833, 0.9999999929870698, 0.9999999953373109, 0.9999999968999163, 0.9999999979388463,
                                  1. + std::numeric_limits<double>::epsilon()})  // clang-format on
      ->setComment(
          "Bins of the PID score histograms. For ROC curves, it is best to generate them as logits : "
          "[0.]+[1/(1+math.exp(-x)) for x in np.linspace(-10, 10, 99)]+[1.+epsilon]");
  descriptions.add("ticlTracksterPIDValidation", desc);
}

DEFINE_FWK_MODULE(TICLTracksterPIDValidation);
