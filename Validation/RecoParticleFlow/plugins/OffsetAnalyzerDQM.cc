#include "DQMOffline/PFTau/interface/Matchers.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include <SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h>

#include <vector>
#include <map>

class OffsetAnalyzerDQM : public DQMEDAnalyzer {
public:
  OffsetAnalyzerDQM(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

protected:
  //Book histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  int getEtaIndex(float eta);

private:
  struct Plot1D {
    std::string name, title, dir;
    int nxbins;
    double xlow, xhigh;
    MonitorElement* plot;

    Plot1D() : name(""), title(""), dir(""), nxbins(0), xlow(0), xhigh(0), plot(nullptr) {}
    Plot1D(const std::string& n, const std::string& t, const std::string& d, int nx, double x0, double x1)
        : name(n), title(t), dir(d), nxbins(nx), xlow(x0), xhigh(x1), plot(nullptr) {}

    virtual void book(DQMStore::IBooker& booker) {
      booker.setCurrentFolder(dir);
      plot = booker.book1D(name, title, nxbins, xlow, xhigh);
    }

    virtual void fill(float value) {
      assert(plot != nullptr);
      plot->Fill(value);
    }

    virtual ~Plot1D() {}
  };

  struct PlotProfile : public Plot1D {
    std::vector<double> xbins;
    int nybins;
    double ylow, yhigh;
    PlotProfile() : Plot1D(), xbins(0), nybins(0), ylow(0), yhigh(0) {}
    PlotProfile(const std::string& n,
                const std::string& t,
                const std::string& d,
                int nx,
                double x0,
                double x1,
                const std::vector<double>& vx,
                int ny,
                double y0,
                double y1)
        : Plot1D(n, t, d, nx, x0, x1), xbins(vx), nybins(ny), ylow(y0), yhigh(y1) {}

    void book(DQMStore::IBooker& booker) override {
      booker.setCurrentFolder(dir);
      plot = booker.bookProfile(name, title, xbins.size() - 1, &xbins[0], nybins, ylow, yhigh, " ");
    }
    //make other booker methods for uniform binning

    void fill2D(double value1, double value2) {
      assert(plot != nullptr);
      plot->Fill(value1, value2);
    }
  };

  std::map<std::string, Plot1D> th1dPlots;
  std::map<std::string, PlotProfile> offsetPlots;
  std::map<int, std::string> pdgMap;

  std::string offsetPlotBaseName;
  std::vector<std::string> pftypes;
  std::vector<double> etabins;

  int muHigh;
  int npvHigh;

  edm::EDGetTokenT<edm::View<reco::Vertex>> pvToken;
  edm::EDGetTokenT<edm::View<PileupSummaryInfo>> muToken;
  edm::EDGetTokenT<edm::View<pat::PackedCandidate>> pfToken;
};

OffsetAnalyzerDQM::OffsetAnalyzerDQM(const edm::ParameterSet& iConfig) {
  offsetPlotBaseName = iConfig.getParameter<std::string>("offsetPlotBaseName");

  pvToken = consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("pvTag"));
  muToken = consumes<edm::View<PileupSummaryInfo>>(iConfig.getParameter<edm::InputTag>("muTag"));
  pfToken = consumes<edm::View<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("pfTag"));

  etabins = iConfig.getParameter<std::vector<double>>("etabins");
  pftypes = iConfig.getParameter<std::vector<std::string>>("pftypes");

  muHigh = iConfig.getUntrackedParameter<int>("muHigh");
  npvHigh = iConfig.getUntrackedParameter<int>("npvHigh");

  //initialize offset plots
  const auto& offset_psets = iConfig.getParameter<std::vector<edm::ParameterSet>>("offsetPlots");
  for (auto& pset : offset_psets) {
    std::string name = pset.getParameter<std::string>("name");
    std::string title = pset.getParameter<std::string>("title");
    std::string dir = pset.getParameter<std::string>("dir");
    std::vector<double> vx = pset.getParameter<std::vector<double>>("vx");
    int ny = pset.getParameter<uint32_t>("ny");
    double y0 = pset.getParameter<double>("y0");
    double y1 = pset.getParameter<double>("y1");

    offsetPlots[name] = PlotProfile(name, title, dir, 0, 0, 0, vx, ny, y0, y1);
  }

  //initialize th1d
  const auto& th1d_psets = iConfig.getParameter<std::vector<edm::ParameterSet>>("th1dPlots");
  for (auto& pset : th1d_psets) {
    std::string name = pset.getParameter<std::string>("name");
    std::string title = pset.getParameter<std::string>("title");
    std::string dir = pset.getParameter<std::string>("dir");
    int nx = pset.getParameter<uint32_t>("nx");
    double x0 = pset.getParameter<double>("x0");
    double x1 = pset.getParameter<double>("x1");

    th1dPlots[name] = Plot1D(name, title, dir, nx, x0, x1);
  }

  //create pdg map
  std::vector<uint32_t> pdgKeys = iConfig.getParameter<std::vector<uint32_t>>("pdgKeys");
  std::vector<std::string> pdgStrs = iConfig.getParameter<std::vector<std::string>>("pdgStrs");
  for (int i = 0, n = pdgKeys.size(); i < n; i++)
    pdgMap[pdgKeys[i]] = pdgStrs[i];
}

void OffsetAnalyzerDQM::bookHistograms(DQMStore::IBooker& booker, edm::Run const&, edm::EventSetup const&) {
  //std::cout << "OffsetAnalyzerDQM booking offset histograms" << std::endl;
  for (auto& pair : offsetPlots) {
    pair.second.book(booker);
  }
  for (auto& pair : th1dPlots) {
    pair.second.book(booker);
  }
}

void OffsetAnalyzerDQM::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  //npv//
  edm::Handle<edm::View<reco::Vertex>> vertexHandle;
  iEvent.getByToken(pvToken, vertexHandle);

  unsigned int nPVall = vertexHandle->size();
  bool isGoodPV[nPVall];
  for (size_t i = 0; i < nPVall; ++i)
    isGoodPV[i] = false;

  int npv = 0;
  for (unsigned int i = 0; i < nPVall; i++) {
    const auto& pv = vertexHandle->at(i);

    if (!pv.isFake() && pv.ndof() >= 4 && fabs(pv.z()) <= 24.0 && fabs(pv.position().rho()) <= 2.0) {
      npv++;
      isGoodPV[i] = true;
    }
  }
  th1dPlots["npv"].fill(npv);
  int npv_in_range = npv;
  if (npv_in_range < 0)
    npv_in_range = 0;
  else if (npv_in_range >= npvHigh)
    npv_in_range = npvHigh - 1;  // make sure int_mu won't lead to non-existing ME

  //mu//
  int int_mu = -1;
  edm::Handle<edm::View<PileupSummaryInfo>> muHandle;
  if (iEvent.getByToken(muToken, muHandle)) {
    const auto& summary = *muHandle;
    auto it = std::find_if(summary.begin(), summary.end(), [](const auto& s) { return s.getBunchCrossing() == 0; });

    if (it->getBunchCrossing() != 0) {
      edm::LogError("OffsetAnalyzerDQM") << "Cannot find the in-time pileup info " << it->getBunchCrossing();
    } else {
      float mu = it->getTrueNumInteractions();
      th1dPlots["mu"].fill(mu);
      int_mu = mu + 0.5;
    }
  }
  if (int_mu >= muHigh)
    int_mu = muHigh - 1;  // make sure int_mu won't lead to non-existing ME

  //create map of pftypes vs total energy / eta
  std::map<std::string, std::vector<double>> m_pftype_etaE;
  int nEta = etabins.size() - 1;
  for (const auto& pftype : pftypes)
    m_pftype_etaE[pftype].assign(nEta, 0.0);

  //pf particles//
  edm::Handle<edm::View<pat::PackedCandidate>> pfHandle;
  iEvent.getByToken(pfToken, pfHandle);

  for (unsigned int i = 0, n = pfHandle->size(); i < n; i++) {
    const auto& cand = pfHandle->at(i);

    int etaIndex = getEtaIndex(cand.eta());
    std::string pftype = pdgMap[abs(cand.pdgId())];
    if (etaIndex == -1 || pftype.empty())
      continue;

    if (pftype == "chm") {  //check charged hadrons ONLY
      bool attached = false;

      for (unsigned int ipv = 0; ipv < nPVall && !attached; ipv++) {
        if (isGoodPV[ipv] && cand.fromPV(ipv) == 3)
          attached = true;  //pv used in fit
      }
      if (!attached)
        pftype = "chu";  //unmatched charged hadron
    }
    ////AOD////
    /*
        reco::TrackRef candTrkRef( cand.trackRef() );
        if ( pftype == "chm" && !candTrkRef.isNull() ) { //check charged hadrons ONLY
            bool attached = false;

            for (auto ipv=vertexHandle->begin(), endpv=vertexHandle->end(); ipv != endpv && !attached; ++ipv) {
                if ( !ipv->isFake() && ipv->ndof() >= 4 && fabs(ipv->z()) < 24 ) { //must be attached to a good pv

                    for(auto ivtrk=ipv->tracks_begin(), endvtrk=ipv->tracks_end(); ivtrk != endvtrk && !attached; ++ivtrk) {
                        reco::TrackRef pvTrkRef(ivtrk->castTo<reco::TrackRef>());
                        if (pvTrkRef == candTrkRef) attached = true;
                    }
                }
            }
            if (!attached) pftype = "chu"; //unmatched charged hadron
        }
*/
    ///////////
    m_pftype_etaE[pftype][etaIndex] += cand.et();
  }

  for (const auto& pair : m_pftype_etaE) {
    std::string pftype = pair.first;
    std::vector<double> etaE = pair.second;

    std::string offset_name_npv = offsetPlotBaseName + "_npv" + std::to_string(npv_in_range) + "_" + pftype;
    if (offsetPlots.find(offset_name_npv) == offsetPlots.end())
      return;  //npv is out of range ()

    for (int i = 0; i < nEta; i++) {
      double eta = 0.5 * (etabins[i] + etabins[i + 1]);
      offsetPlots[offset_name_npv].fill2D(eta, etaE[i]);
    }

    if (int_mu != -1) {
      std::string offset_name_mu = offsetPlotBaseName + "_mu" + std::to_string(int_mu) + "_" + pftype;
      if (offsetPlots.find(offset_name_mu) == offsetPlots.end())
        return;  //mu is out of range

      for (int i = 0; i < nEta; i++) {
        double eta = 0.5 * (etabins[i] + etabins[i + 1]);
        offsetPlots[offset_name_mu].fill2D(eta, etaE[i]);
      }
    }
  }
}

int OffsetAnalyzerDQM::getEtaIndex(float eta) {
  int nEta = etabins.size() - 1;

  for (int i = 0; i < nEta; i++) {
    if (etabins[i] <= eta && eta < etabins[i + 1])
      return i;
  }
  if (eta == etabins[nEta])
    return nEta - 1;
  else
    return -1;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OffsetAnalyzerDQM);
