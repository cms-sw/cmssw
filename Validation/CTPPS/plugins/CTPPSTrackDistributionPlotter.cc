/****************************************************************************
 *
 * This is a part of CTPPS validation software
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "TFile.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TGraph.h"

#include <map>
#include <memory>

        
//----------------------------------------------------------------------------------------------------

class CTPPSTrackDistributionPlotter : public edm::one::EDAnalyzer<> {
public:
  explicit CTPPSTrackDistributionPlotter(const edm::ParameterSet&);

  ~CTPPSTrackDistributionPlotter() override {}

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> tracksToken_;

  double x_pitch_pixels_;

  std::string outputFile_;

  unsigned int events_total_;
  std::map<unsigned int, unsigned int> events_per_arm_;

  struct RPPlots {
    bool initialized;

    std::unique_ptr<TH2D> h2_y_vs_x;
    std::unique_ptr<TProfile> p_y_vs_x;
    std::unique_ptr<TH1D> h_x;
    std::unique_ptr<TH1D> h_y;
    std::unique_ptr<TH1D> h_time;

    RPPlots() : initialized(false) {}

    void init(bool pixel, double pitch) {
      const double bin_size_x = (pixel) ? pitch * cos(18.4 / 180. * M_PI) : 100E-3;

      h2_y_vs_x = std::make_unique<TH2D>("", "", 300, -10., +70., 600, -30., +30.);
      p_y_vs_x = std::make_unique<TProfile>("", "", 300, -10., +70.);

      int n_mi = ceil(10. / bin_size_x);
      int n_pl = ceil(70. / bin_size_x);

      h_x = std::make_unique<TH1D>("", "", n_mi + n_pl, -n_mi * bin_size_x, +n_pl * bin_size_x);

      h_y = std::make_unique<TH1D>("", "", 300, -15., +15.);

      h_time = std::make_unique<TH1D>("", ";time", 500, -50., +50.);

      initialized = true;
    }

    void fill(double x, double y, double time) {
      h2_y_vs_x->Fill(x, y);
      p_y_vs_x->Fill(x, y);
      h_x->Fill(x);
      h_y->Fill(y);
      h_time->Fill(time);
    }

    void write() const {
      h2_y_vs_x->Write("h2_y_vs_x");
      p_y_vs_x->Write("p_y_vs_x");
      h_x->Write("h_x");
      h_y->Write("h_y");
      h_time->Write("h_time");
    }
  };

  std::map<unsigned int, RPPlots> rpPlots;

  struct ArmPlots {
    unsigned int rpId_N, rpId_F;

    std::unique_ptr<TH1D> h_de_x, h_de_y;
    std::unique_ptr<TProfile> p_de_x_vs_x, p_de_y_vs_x;
    std::unique_ptr<TProfile2D> p2_de_x_vs_x_y, p2_de_y_vs_x_y;
    std::unique_ptr<TH2D> h2_de_x_vs_x, h2_de_y_vs_x;
    std::unique_ptr<TH2D> h2_de_y_vs_de_x;

    struct Stat {
      unsigned int sN = 0, s1 = 0;
    };

    std::map<unsigned int, std::map<unsigned int, Stat>> m_stat;

    ArmPlots()
        : h_de_x(new TH1D("", ";x^{F} - x^{N}", 100, -1., +1.)),
          h_de_y(new TH1D("", ";y^{F} - y^{N}", 100, -1., +1.)),
          p_de_x_vs_x(new TProfile("", ";x^{N};x^{F} - x^{N}", 40, 0., 40.)),
          p_de_y_vs_x(new TProfile("", ";x^{N};y^{F} - y^{N}", 40, 0., 40.)),
          p2_de_x_vs_x_y(new TProfile2D("", ";x;y", 40, 0., 40., 40, -20., +20.)),
          p2_de_y_vs_x_y(new TProfile2D("", ";x;y", 40, 0., 40., 40, -20., +20.)),
          h2_de_x_vs_x(new TH2D("", ";x^{N};x^{F} - x^{N}", 80, 0., 40., 100, -1., +1.)),
          h2_de_y_vs_x(new TH2D("", ";x^{N};y^{F} - y^{N}", 80, 0., 40., 100, -1., +1.)),
          h2_de_y_vs_de_x(new TH2D("", ";x^{F} - x^{N};y^{F} - y^{N}", 100, -1., +1., 100, -1., +1.)) {}

    void fill(double x_N, double y_N, double x_F, double y_F) {
      h_de_x->Fill(x_F - x_N);
      h_de_y->Fill(y_F - y_N);

      p_de_x_vs_x->Fill(x_N, x_F - x_N);
      p_de_y_vs_x->Fill(x_N, y_F - y_N);

      p2_de_x_vs_x_y->Fill(x_N, y_N, x_F - x_N);
      p2_de_y_vs_x_y->Fill(x_N, y_N, y_F - y_N);

      h2_de_x_vs_x->Fill(x_N, x_F - x_N);
      h2_de_y_vs_x->Fill(x_N, y_F - y_N);

      h2_de_y_vs_de_x->Fill(x_F - x_N, y_F - y_N);
    }

    void write() const {
      h_de_x->Write("h_de_x");
      h_de_y->Write("h_de_y");

      p_de_x_vs_x->Write("p_de_x_vs_x");
      p_de_y_vs_x->Write("p_de_y_vs_x");

      p2_de_x_vs_x_y->Write("p2_de_x_vs_x_y");
      p2_de_y_vs_x_y->Write("p2_de_y_vs_x_y");

      h2_de_x_vs_x->Write("h2_de_x_vs_x");
      h2_de_y_vs_x->Write("h2_de_y_vs_x");

      h2_de_y_vs_de_x->Write("h2_de_y_vs_de_x");

      for (const auto& rp : m_stat) {
        TGraph* g = new TGraph();

        char buf[100];
        sprintf(buf, "g_mean_track_mult_run_%u", rp.first);
        g->SetName(buf);

        for (const auto& lsp : rp.second) {
          const int idx = g->GetN();
          const double m = (lsp.second.s1 > 0) ? double(lsp.second.sN) / lsp.second.s1 : 0.;
          g->SetPoint(idx, lsp.first, m);
        }

        g->Write();
      }
    }
  };

  std::map<unsigned int, ArmPlots> armPlots;
};

//----------------------------------------------------------------------------------------------------

CTPPSTrackDistributionPlotter::CTPPSTrackDistributionPlotter(const edm::ParameterSet& iConfig)
    : tracksToken_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("tagTracks"))),
      x_pitch_pixels_(iConfig.getUntrackedParameter<double>("x_pitch_pixels", 150E-3)),
      outputFile_(iConfig.getParameter<std::string>("outputFile")),
      events_total_(0) {
  armPlots[0].rpId_N = iConfig.getParameter<unsigned int>("rpId_45_N");
  armPlots[0].rpId_F = iConfig.getParameter<unsigned int>("rpId_45_F");

  armPlots[1].rpId_N = iConfig.getParameter<unsigned int>("rpId_56_N");
  armPlots[1].rpId_F = iConfig.getParameter<unsigned int>("rpId_56_F");
}

//----------------------------------------------------------------------------------------------------

void CTPPSTrackDistributionPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  // get input
  edm::Handle<CTPPSLocalTrackLiteCollection> tracks;
  iEvent.getByToken(tracksToken_, tracks);

  // process tracks
  std::map<unsigned int, unsigned int> m_mult;

  for (const auto& trk : *tracks) {
    CTPPSDetId rpId(trk.rpId());
    unsigned int rpDecId = rpId.arm() * 100 + rpId.station() * 10 + rpId.rp();
    bool rpPixel = (rpId.subdetId() == CTPPSDetId::sdTrackingPixel);

    auto& pl = rpPlots[rpDecId];
    if (!pl.initialized)
      pl.init(rpPixel, x_pitch_pixels_);

    pl.fill(trk.x(), trk.y(), trk.time());

    m_mult[rpId.arm()]++;
  }

  for (unsigned int arm = 0; arm < 2; ++arm) {
    auto& st = armPlots[arm].m_stat[iEvent.id().run()][iEvent.id().luminosityBlock()];
    st.s1++;
    st.sN += m_mult[arm];
  }

  for (const auto& t1 : *tracks) {
    const CTPPSDetId rpId1(t1.rpId());

    for (const auto& t2 : *tracks) {
      const CTPPSDetId rpId2(t2.rpId());

      if (rpId1.arm() != rpId2.arm())
        continue;

      auto& ap = armPlots[rpId1.arm()];

      const unsigned int rpDecId1 = rpId1.arm() * 100 + rpId1.station() * 10 + rpId1.rp();
      const unsigned int rpDecId2 = rpId2.arm() * 100 + rpId2.station() * 10 + rpId2.rp();

      if (rpDecId1 == ap.rpId_N && rpDecId2 == ap.rpId_F)
        ap.fill(t1.x(), t1.y(), t2.x(), t2.y());
    }
  }

  // update counters
  events_total_++;

  if (m_mult[0] > 0)
    events_per_arm_[0]++;
  if (m_mult[1] > 0)
    events_per_arm_[1]++;
}

//----------------------------------------------------------------------------------------------------

void CTPPSTrackDistributionPlotter::endJob() {
  edm::LogInfo("CTPPSTrackDistributionPlotter")
      << "    events processed: " << events_total_ << " (" << std::scientific << double(events_total_) << ")\n"
      << "    events with tracks in sector 45: " << events_per_arm_[0] << " (" << double(events_per_arm_[0]) << ")\n"
      << "    events with tracks in sector 56: " << events_per_arm_[1] << " (" << double(events_per_arm_[1]) << ")";

  auto f_out = std::make_unique<TFile>(outputFile_.c_str(), "recreate");

  for (const auto& it : rpPlots) {
    gDirectory = f_out->mkdir(Form("RP %u", it.first));
    it.second.write();
  }

  for (const auto& it : armPlots) {
    gDirectory = f_out->mkdir(Form("arm %u", it.first));
    it.second.write();
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSTrackDistributionPlotter);
