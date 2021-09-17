#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include <TH2F.h>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

class HcalSimHitAnalysis : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  HcalSimHitAnalysis(const edm::ParameterSet& ps);
  ~HcalSimHitAnalysis() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  static const int ndets_ = 4;
  std::string g4Label_, hcalHits_;
  bool verbose_, testNumber_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_calo_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_HRNDC_;
  TH2F *poszp_[ndets_], *poszn_[ndets_];
};

HcalSimHitAnalysis::HcalSimHitAnalysis(const edm::ParameterSet& ps) {
  usesResource(TFileService::kSharedResource);

  g4Label_ = ps.getUntrackedParameter<std::string>("moduleLabel", "g4SimHits");
  hcalHits_ = ps.getUntrackedParameter<std::string>("HitCollection", "HcalHits");
  verbose_ = ps.getUntrackedParameter<bool>("Verbose", false);
  testNumber_ = ps.getUntrackedParameter<bool>("TestNumber", true);

  tok_calo_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hcalHits_));
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_HRNDC_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord>();

  edm::LogVerbatim("HitStudy") << "Module Label: " << g4Label_ << "   Hits: " << hcalHits_ << " testNumber "
                               << testNumber_;
}

void HcalSimHitAnalysis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("ModuleLabel", "g4SimHits");
  desc.addUntracked<std::string>("HitCollection", "HcalHits");
  desc.addUntracked<bool>("Verbose", false);
  desc.addUntracked<bool>("TestNumber", true);
  descriptions.add("hcalSimHitAnalysis", desc);
}

void HcalSimHitAnalysis::beginJob() {
  edm::Service<TFileService> tfile;
  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  char name[20], title[120];
  std::string dets[ndets_] = {"HB", "HE", "HO", "HF"};
  int nx[ndets_] = {100, 100, 350, 160};
  double xlo[ndets_] = {0, -300, 0, -160};
  double xhi[ndets_] = {500, 300, 3500, 160};
  int ny[ndets_] = {100, 100, 50, 160};
  double ylo[ndets_] = {170, -300, 375, -160};
  double yhi[ndets_] = {370, 300, 425, 160};
  std::string xttl[ndets_] = {"|z| (cm)", "x (cm)", "|z| (cm)", "x (cm)"};
  std::string yttl[ndets_] = {"#rho (cm)", "y (cm)", "#rho (cm)", "y (cm)"};
  for (int i = 0; i < ndets_; i++) {
    sprintf(name, "poszp%d", i);
    sprintf(title, "%s+", dets[i].c_str());
    poszp_[i] = tfile->make<TH2F>(name, title, nx[i], xlo[i], xhi[i], ny[i], ylo[i], yhi[i]);
    poszp_[i]->GetXaxis()->SetTitle(xttl[i].c_str());
    poszp_[i]->GetYaxis()->SetTitle(yttl[i].c_str());
    sprintf(title, "%s-", dets[i].c_str());
    poszp_[i]->GetYaxis()->SetTitleOffset(1.2);
    poszp_[i]->Sumw2();
    sprintf(name, "poszn%d", i);
    poszn_[i] = tfile->make<TH2F>(name, title, nx[i], xlo[i], xhi[i], ny[i], ylo[i], yhi[i]);
    poszn_[i]->GetXaxis()->SetTitle(xttl[i].c_str());
    poszn_[i]->GetYaxis()->SetTitle(yttl[i].c_str());
    poszn_[i]->GetYaxis()->SetTitleOffset(1.2);
    poszn_[i]->Sumw2();
  }
}

void HcalSimHitAnalysis::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  if (verbose_)
    edm::LogVerbatim("HitStudy") << "Run = " << e.id().run() << " Event = " << e.id().event();

  // get hcalGeometry
  const CaloGeometry* geo = &iS.getData(tok_geom_);
  const HcalGeometry* hgeom = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  const auto& pHRNDC = iS.getData(tok_HRNDC_);
  const HcalDDDRecConstants* hcons = &pHRNDC;

  edm::Handle<edm::PCaloHitContainer> hitsCalo;
  e.getByToken(tok_calo_, hitsCalo);
  bool getHits = (hitsCalo.isValid());
  uint32_t nhits = (getHits) ? hitsCalo->size() : 0;
  if (verbose_)
    edm::LogVerbatim("HitStudy") << "HcalSimHitAnalysis: Input flags Hits " << getHits << " with " << nhits << " hits";
  if (getHits) {
    std::vector<PCaloHit> hits;
    hits.insert(hits.end(), hitsCalo->begin(), hitsCalo->end());
    if (!hits.empty()) {
      std::map<HcalDetId, double> hitMap;
      for (auto hit : hits) {
        double edep = hit.energy();
        uint32_t id = hit.id();
        HcalDetId hid = (testNumber_) ? HcalHitRelabeller::relabel(id, hcons) : HcalDetId(id);
        auto it = hitMap.find(hid);
        if (it == hitMap.end()) {
          hitMap[hid] = edep;
        } else {
          (it->second) += edep;
        }
      }

      for (auto it : hitMap) {
        HcalDetId id(it.first);
        GlobalPoint gpos = hgeom->getPosition(id);
        HcalSubdetector subdet = (id).subdet();
        int indx =
            ((subdet == HcalBarrel)
                 ? 0
                 : ((subdet == HcalEndcap) ? 1 : ((subdet == HcalOuter) ? 2 : ((subdet == HcalForward) ? 3 : -1))));
        if (verbose_)
          edm::LogVerbatim("HitStudy") << "HcalSimHitAnalysis: " << id << ":" << it.second << " at " << gpos
                                       << " subdet " << subdet << ":" << indx;
        if (indx >= 0) {
          double x = ((indx == 0) || (indx == 2)) ? std::abs(gpos.z()) : gpos.x();
          double y = ((indx == 0) || (indx == 2)) ? (gpos.perp()) : gpos.y();
          if (id.zside() >= 0)
            poszp_[indx]->Fill(x, y);
          else
            poszn_[indx]->Fill(x, y);
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalSimHitAnalysis);
