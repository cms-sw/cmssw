// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "SimCalorimetry/HGCalSimAlgos/interface/HGCalSciNoiseMap.h"

//ROOT headers
#include <TH2F.h>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//STL headers
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <set>

#include "vdt/vdtMath.h"

//
// class declaration
//

class HGCHEbackSignalScalerAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HGCHEbackSignalScalerAnalyzer(const edm::ParameterSet&);
  ~HGCHEbackSignalScalerAnalyzer() override;

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  void createBinning(const std::vector<DetId>&);
  void printBoundaries();

  // ----------member data ---------------------------
  edm::Service<TFileService> fs;

  std::string doseMap_;
  uint32_t doseMapAlgo_;
  std::string sipmMap_;
  double refIdark_;

  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;

  std::set<int> allLayers_, allIphi_, allIeta_;

  const HGCalGeometry* gHGCal_;
  const HGCalDDDConstants* hgcCons_;

  std::map<int, std::map<TString, TH1F*> > layerHistos_;
  std::map<TString, TH2F*> histos_;
};

//
// constructors and destructor
//
HGCHEbackSignalScalerAnalyzer::HGCHEbackSignalScalerAnalyzer(const edm::ParameterSet& iConfig)
    : doseMap_(iConfig.getParameter<std::string>("doseMap")),
      doseMapAlgo_(iConfig.getParameter<uint32_t>("doseMapAlgo")),
      sipmMap_(iConfig.getParameter<std::string>("sipmMap")),
      refIdark_(iConfig.getParameter<double>("referenceIdark")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag("", "HGCalHEScintillatorSensitive"))),
      dddToken_(
          esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag("", "HGCalHEScintillatorSensitive"))) {
  usesResource("TFileService");
  fs->file().cd();
}

HGCHEbackSignalScalerAnalyzer::~HGCHEbackSignalScalerAnalyzer() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
void HGCHEbackSignalScalerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //get geometry
  const auto& geomhandle = iSetup.getHandle(geomToken_);
  if (!geomhandle.isValid()) {
    edm::LogError("HGCHEbackSignalScalerAnalyzer") << "Cannot get valid HGCalGeometry Object";
    return;
  }
  gHGCal_ = geomhandle.product();
  const std::vector<DetId>& detIdVec = gHGCal_->getValidDetIds();
  LogDebug("HGCHEbackSignalScalerAnalyzer") << "Total number of DetIDs: " << detIdVec.size();

  //get ddd constants
  const auto& dddhandle = iSetup.getHandle(dddToken_);
  if (!dddhandle.isValid()) {
    edm::LogError("HGCHEbackSignalScalerAnalyzer") << "Cannot initiate HGCalDDDConstants";
    return;
  }
  hgcCons_ = dddhandle.product();

  //setup maps
  createBinning(detIdVec);
  int minLayer(*std::min_element(allLayers_.begin(), allLayers_.end()));
  int maxLayer(*std::max_element(allLayers_.begin(), allLayers_.end()));
  int nLayers(maxLayer - minLayer + 1);
  int minIeta(*std::min_element(allIeta_.begin(), allIeta_.end()));
  int maxIeta(*std::max_element(allIeta_.begin(), allIeta_.end()));
  int nIeta(maxIeta - minIeta + 1);
  int minIphi(*std::min_element(allIphi_.begin(), allIphi_.end()));
  int maxIphi(*std::max_element(allIphi_.begin(), allIphi_.end()));
  int nIphi(maxIphi - minIphi + 1);

  TString hnames[] = {"count", "radius", "dose", "fluence", "s", "n", "sn", "lysf", "gain", "thr"};
  TString htitles[] = {"tiles",
                       "#rho [cm]",
                       "<Dose> [krad]",
                       "<Fluence> [1 MeV n_{eq}/cm^{2}]",
                       "<S> [pe]",
                       "<#sigma_{N}> [pe]",
                       "<S/#sigma_{N}>",
                       "LY SF",
                       "<Gain>",
                       "<Threshold> [ADC]"};
  for (size_t i = 0; i < sizeof(hnames) / sizeof(TString); i++) {
    histos_[hnames[i] + "_ieta"] = fs->make<TH2F>(
        hnames[i] + "_ieta", ";Layer;i-#eta;" + htitles[i], nLayers, minLayer, maxLayer + 1, nIeta, minIeta, maxIeta);
    histos_[hnames[i] + "_iphi"] = fs->make<TH2F>(
        hnames[i] + "_iphi", ";Layer;i-#phi;" + htitles[i], nLayers, minLayer, maxLayer + 1, nIphi, minIphi, maxIphi);

    //add per-layer profiles
    for (int ilay = minLayer; ilay <= maxLayer; ilay++) {
      if (layerHistos_.count(ilay) == 0) {
        std::map<TString, TH1F*> templ_map;
        layerHistos_[ilay] = templ_map;
      }
      TString hnameLay(Form("%s_lay%d_", hnames[i].Data(), ilay));
      layerHistos_[ilay][hnames[i] + "_ieta"] =
          fs->make<TH1F>(hnameLay + "ieta", ";i-#eta;" + htitles[i], nIeta, minIeta, maxIeta);
      layerHistos_[ilay][hnames[i] + "_iphi"] =
          fs->make<TH1F>(hnameLay + "iphi", ";i-#phi;" + htitles[i], nIphi, minIphi, maxIphi);
    }
  }

  //instantiate scaler
  HGCalSciNoiseMap scal;
  scal.setDoseMap(doseMap_, doseMapAlgo_);
  scal.setReferenceDarkCurrent(refIdark_);
  scal.setSipmMap(sipmMap_);
  scal.setGeometry(gHGCal_);

  //loop over valid detId from the HGCHEback
  for (auto myId : detIdVec) {
    HGCScintillatorDetId scId(myId.rawId());

    int layer = std::abs(scId.layer());
    std::pair<int, int> ietaphi = scId.ietaphi();
    int ieta = std::abs(ietaphi.first);
    int iphi = std::abs(ietaphi.second);

    //characteristics of this tile
    GlobalPoint global = gHGCal_->getPosition(scId);
    double radius = std::sqrt(std::pow(global.x(), 2) + std::pow(global.y(), 2));
    double dose = scal.getDoseValue(DetId::HGCalHSc, layer, radius);
    double fluence = scal.getFluenceValue(DetId::HGCalHSc, layer, radius);
    auto opChar = scal.scaleByDose(scId, radius);
    float s = opChar.s;
    float n = opChar.n;
    float sn = n > 0 ? s / n : -1.;
    float lysf = opChar.lySF;
    int gain = opChar.gain;
    int thrADC = opChar.thrADC;

    histos_["count_ieta"]->Fill(layer, ieta, 1);
    histos_["count_iphi"]->Fill(layer, iphi, 1);
    histos_["radius_ieta"]->Fill(layer, ieta, radius);
    histos_["radius_iphi"]->Fill(layer, iphi, radius);
    histos_["dose_ieta"]->Fill(layer, ieta, dose);
    histos_["dose_iphi"]->Fill(layer, iphi, dose);
    histos_["fluence_ieta"]->Fill(layer, ieta, fluence);
    histos_["fluence_iphi"]->Fill(layer, iphi, fluence);
    histos_["s_ieta"]->Fill(layer, ieta, s);
    histos_["s_iphi"]->Fill(layer, iphi, s);
    histos_["n_ieta"]->Fill(layer, ieta, n);
    histos_["n_iphi"]->Fill(layer, iphi, n);
    histos_["sn_ieta"]->Fill(layer, ieta, sn);
    histos_["sn_iphi"]->Fill(layer, iphi, sn);
    histos_["lysf_ieta"]->Fill(layer, ieta, lysf);
    histos_["lysf_iphi"]->Fill(layer, iphi, lysf);
    histos_["gain_ieta"]->Fill(layer, ieta, gain);
    histos_["gain_iphi"]->Fill(layer, iphi, gain);
    histos_["thr_ieta"]->Fill(layer, ieta, thrADC);
    histos_["thr_iphi"]->Fill(layer, iphi, thrADC);

    if (layerHistos_.count(layer) == 0) {
      continue;
    }

    //per layer
    layerHistos_[layer]["count_ieta"]->Fill(ieta, 1);
    layerHistos_[layer]["count_iphi"]->Fill(iphi, 1);
    layerHistos_[layer]["radius_ieta"]->Fill(ieta, radius);
    layerHistos_[layer]["radius_iphi"]->Fill(iphi, radius);
    layerHistos_[layer]["dose_ieta"]->Fill(ieta, dose);
    layerHistos_[layer]["dose_iphi"]->Fill(iphi, dose);
    layerHistos_[layer]["fluence_ieta"]->Fill(ieta, fluence);
    layerHistos_[layer]["fluence_iphi"]->Fill(iphi, fluence);
    layerHistos_[layer]["s_ieta"]->Fill(ieta, s);
    layerHistos_[layer]["s_iphi"]->Fill(iphi, s);
    layerHistos_[layer]["n_ieta"]->Fill(ieta, n);
    layerHistos_[layer]["n_iphi"]->Fill(iphi, n);
    layerHistos_[layer]["sn_ieta"]->Fill(ieta, sn);
    layerHistos_[layer]["sn_iphi"]->Fill(iphi, sn);
    layerHistos_[layer]["lysf_ieta"]->Fill(ieta, lysf);
    layerHistos_[layer]["lysf_iphi"]->Fill(iphi, lysf);
    layerHistos_[layer]["gain_ieta"]->Fill(ieta, gain);
    layerHistos_[layer]["gain_iphi"]->Fill(iphi, gain);
    layerHistos_[layer]["thr_ieta"]->Fill(ieta, thrADC);
    layerHistos_[layer]["thr_iphi"]->Fill(iphi, thrADC);
  }

  //normalize by the number of tiles to have the mean stored
  for (auto h : histos_) {
    TString name(h.first);
    if (name.Contains("count_"))
      continue;
    TString pfix(name.Contains("ieta") ? "ieta" : "iphi");
    h.second->Divide(histos_["count_" + pfix]);
  }

  for (auto h : layerHistos_) {
    int lay(h.first);
    for (auto hh : layerHistos_[lay]) {
      TString name(hh.first);
      if (name.Contains("count_"))
        continue;
      TString pfix(name.Contains("ieta") ? "ieta" : "iphi");
      hh.second->Divide(layerHistos_[lay]["count_" + pfix]);
    }
  }
}

//
void HGCHEbackSignalScalerAnalyzer::createBinning(const std::vector<DetId>& detIdVec) {
  //loop over detids to find possible values
  for (auto myId : detIdVec) {
    HGCScintillatorDetId scId(myId.rawId());
    int layer = std::abs(scId.layer());
    std::pair<int, int> ietaphi = scId.ietaphi();

    allLayers_.insert(std::abs(layer));
    allIeta_.insert(std::abs(ietaphi.first));
    allIphi_.insert(std::abs(ietaphi.second));
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCHEbackSignalScalerAnalyzer);
