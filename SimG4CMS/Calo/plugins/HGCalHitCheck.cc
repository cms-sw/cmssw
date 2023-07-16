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

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <map>
#include <string>
#include <vector>
#include <TH1D.h>

class HGcalHitCheck : public edm::one::EDAnalyzer<> {
public:
  HGcalHitCheck(const edm::ParameterSet& ps);
  ~HGcalHitCheck() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override;
  void endJob() override {}

private:
  const std::string g4Label_, caloHitSource_, nameSense_, nameDetector_, tag_;
  const int layers_, verbosity_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_calo_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  bool histos_;
  TH1D *h_hits_,  *h_hit1_, *h_hit2_;
  std::vector<TH1D*> h_hitL_, h_hitF_, h_hitP_;
};

HGcalHitCheck::HGcalHitCheck(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("moduleLabel")),
      caloHitSource_(ps.getParameter<std::string>("caloHitSource")),
      nameSense_(ps.getParameter<std::string>("nameSense")),
      nameDetector_(ps.getParameter<std::string>("tag")),
      tag_(ps.getParameter<std::string>("nameDevice")),
      layers_(ps.getParameter<int>("layers")),
      verbosity_(ps.getParameter<int>("verbosity")),
      tok_calo_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, caloHitSource_))),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})),
      histos_(false) {
  edm::LogVerbatim("HitStudy") << "Test Hit ID for " << nameDetector_ << " using SimHits for " << nameSense_
                               << " with module Label: " << g4Label_ << "   Hits: " << caloHitSource_;
}

void HGcalHitCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("moduleLabel", "g4SimHits");
  desc.add<std::string>("caloHitSource", "HGCHitsEE");
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  desc.add<std::string>("nameDevice", "HGCal EE");
  desc.add<std::string>("tag","DDD");
  desc.add<int>("layers", 26);
  desc.add<int>("verbosity", 0);
  descriptions.add("hgcalHitCheckEE", desc);
}

void HGcalHitCheck::beginJob() {
  edm::Service<TFileService> fs;
  if (!fs.isAvailable()) {
    edm::LogVerbatim("HitStudy") << "TFileService unavailable: no histograms";
  } else {
    histos_ = true;
    char name[100], title[200];
    sprintf (name, "HitsL");
    sprintf (title, "Number of hits in %s for %s", nameSense_.c_str(), tag_.c_str());
    h_hits_ = fs->make<TH1D>(name, title, 1000, 0, 5000.);
    h_hits_->GetXaxis()->SetTitle(title);
    h_hits_->GetYaxis()->SetTitle("Hits");
    h_hits_->Sumw2();
    sprintf (name, "HitsF");
    sprintf (title, "Number of hits in %s for %s in Full Wafers or SiPM 2", nameSense_.c_str(), tag_.c_str());
    h_hit1_ = fs->make<TH1D>(name, title, 1000, 0, 5000.);
    h_hit1_->GetXaxis()->SetTitle(title);
    h_hit1_->GetYaxis()->SetTitle("Hits");
    h_hit1_->Sumw2();
    sprintf (name, "HitsP");
    sprintf (title, "Number of hits in %s for %s in Partial Wafers or SiPM 4", nameSense_.c_str(), tag_.c_str());
    h_hit2_ = fs->make<TH1D>(name, title, 1000, 0, 5000.);
    h_hit2_->GetXaxis()->SetTitle(title);
    h_hit2_->GetYaxis()->SetTitle("Hits");
    h_hit2_->Sumw2();
    for (int k = 0; k < layers_; ++k) {
      sprintf (name, "HitsL%d", k + 1);
      sprintf (title, "Number of hits in %s for %s in Layer %d", nameSense_.c_str(), tag_.c_str(), k + 1);
      h_hitL_.emplace_back(fs->make<TH1D>(name, title, 1000, 0, 5000.));
      h_hitL_.back()->GetXaxis()->SetTitle(title);
      h_hitL_.back()->GetYaxis()->SetTitle("Hits");
      h_hitL_.back()->Sumw2();
      sprintf (name, "HitsF%d", k + 1);
      sprintf (title, "Number of hits in %s for %s in Full Wafers or SiPM 2 of Layer %d", nameSense_.c_str(), tag_.c_str(), k + 1);
      h_hitF_.emplace_back(fs->make<TH1D>(name, title, 1000, 0, 5000.));
      h_hitF_.back()->GetXaxis()->SetTitle(title);
      h_hitF_.back()->GetYaxis()->SetTitle("Hits");
      h_hitF_.back()->Sumw2();
      sprintf (name, "HitsP%d", k + 1);
      sprintf (title, "Number of hits in %s for %s in Partial Wafers or SiPM 4  of Layer %d", nameSense_.c_str(), tag_.c_str(), k + 1);
      h_hitP_.emplace_back(fs->make<TH1D>(name, title, 1000, 0, 5000.));
      h_hitP_.back()->GetXaxis()->SetTitle(title);
      h_hitP_.back()->GetYaxis()->SetTitle("Hits");
      h_hitP_.back()->Sumw2();
    }
  }
}

void HGcalHitCheck::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  if (verbosity_ > 0)
    edm::LogVerbatim("HitStudy") << "Run = " << e.id().run() << " Event = " << e.id().event();

  // get hcalGeometry
  const HGCalGeometry* geom = &iS.getData(geomToken_);
  const HGCalDDDConstants& hgc = geom->topology().dddConstants();
  const std::vector<DetId>& validIds = geom->getValidDetIds();
  edm::LogVerbatim("HitStudy") << "Detector " << nameSense_ << " with " << validIds.size() << " valid cells";

  const edm::Handle<edm::PCaloHitContainer>& hitsCalo = e.getHandle(tok_calo_);
  bool getHits = (hitsCalo.isValid());
  uint32_t nhits = (getHits) ? hitsCalo->size() : 0;
  uint32_t wafer(0), tiles(0);
  if (verbosity_ > 1)
    edm::LogVerbatim("HitStudy") << "HGcalHitCheck: Input flags Hits " << getHits << " with " << nhits << " hits";
  if (histos_)
    h_hits_->Fill(nhits);

  if (getHits) {
    std::vector<PCaloHit> hits;
    hits.insert(hits.end(), hitsCalo->begin(), hitsCalo->end());
    if (!hits.empty()) {
      for (auto hit : hits) {
	if (histos_) {
	  if ((nameSense_ == "HGCalEESensitive") || (nameSense_ == "HGCalHESiliconSensitive")) {
	    ++wafer;
	    HGCSiliconDetId id(hit.id());
	    int lay = id.layer();
	    h_hitL_[lay-1]->Fill(nhits);
	    HGCalParameters::waferInfo info = hgc.waferInfo(lay, id.waferU(), id.waferV());
	    if (info.part == HGCalTypes::WaferFull) {
	      h_hit1_->Fill(nhits);
	      h_hitF_[lay-1]->Fill(nhits);
	    } else {
	      h_hit2_->Fill(nhits);
	      h_hitP_[lay-1]->Fill(nhits);
	    }
	  } else {
	    ++tiles;
	    HGCScintillatorDetId id(hit.id());
	    int lay = id.layer();
	    h_hitL_[lay-1]->Fill(nhits);
	    int sipm = id.sipm();
	    if (sipm == 1) {
	      h_hit2_->Fill(nhits);
	      h_hitP_[lay-1]->Fill(nhits);
	    } else {
	      h_hit1_->Fill(nhits);
	      h_hitF_[lay-1]->Fill(nhits);
	    }
          }
        }
      }
    }
  }
  edm::LogVerbatim("HitStudy") << "Total hits = " << nhits << " Wafer DetIds = " << wafer << " Tile DetIds = " << tiles;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGcalHitCheck);
