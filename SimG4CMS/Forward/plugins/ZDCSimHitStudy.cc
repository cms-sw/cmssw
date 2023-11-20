#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <CLHEP/Units/GlobalPhysicalConstants.h>
#include <CLHEP/Units/GlobalSystemOfUnits.h>

#include <TH1F.h>

#include <string>
#include <vector>

class ZDCSimHitStudy : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  ZDCSimHitStudy(const edm::ParameterSet &ps);
  ~ZDCSimHitStudy() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void beginJob() override;
  void endJob() override {}
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  void analyzeHits(const std::vector<PCaloHit> &);

private:
  const std::string g4Label_;
  const std::string hitLab_;
  const double maxEnergy_, tCut_;
  const bool verbose_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> toks_calo_;
  TH1F *hit_, *eTot_, *eTotT_, *edep_, *time_, *indx_;
};

ZDCSimHitStudy::ZDCSimHitStudy(const edm::ParameterSet &ps)
    : g4Label_(ps.getParameter<std::string>("ModuleLabel")),
      hitLab_(ps.getParameter<std::string>("HitCollection")),
      maxEnergy_(ps.getParameter<double>("MaxEnergy")),
      tCut_(ps.getParameter<double>("TimeCut")),
      verbose_(ps.getParameter<bool>("Verbose")),
      toks_calo_(consumes<edm::PCaloHitContainer>(edm::InputTag{g4Label_, hitLab_})) {
  usesResource(TFileService::kSharedResource);

  edm::LogVerbatim("HitStudy") << "HOSimHitStudy::Module Label: " << g4Label_ << "   Hits: " << hitLab_
                               << "   MaxEnergy: " << maxEnergy_ << " time Cut " << tCut_;
}

void ZDCSimHitStudy::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ModuleLabel", "g4SimHits");
  desc.add<std::string>("HitCollection", "ZDCHITS");
  desc.add<double>("MaxEnergy", 50.0);
  desc.add<double>("TimeCut", 2000.0);
  desc.add<bool>("Verbose", false);
  descriptions.add("zdcSimHitStudy", desc);
}

void ZDCSimHitStudy::beginJob() {
  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  double ymax = maxEnergy_;
  hit_ = tfile->make<TH1F>("Hits", "Number of Hits", 100, 0., 100);
  edep_ = tfile->make<TH1F>("Edep", "Deposited Energy (GeV)", 1000, 0., ymax);
  eTot_ = tfile->make<TH1F>("ETot", "Total Energy in a time window (GeV)", 1000, 0., ymax);
  eTotT_ = tfile->make<TH1F>("ETotT", "Total Energy (GeV)", 1000, 0., ymax);
  time_ = tfile->make<TH1F>("Time", "Hit Time (ns)", 2000, 0., 2000);
  indx_ = tfile->make<TH1F>("Indx", "Hit ID", 100, 0., 100);
}

void ZDCSimHitStudy::analyze(const edm::Event &e, const edm::EventSetup &) {
  edm::LogVerbatim("HitStudy") << "ZDCSimHitStudy::Run = " << e.id().run() << " Event = " << e.id().event();

  const edm::Handle<edm::PCaloHitContainer> &hitsCalo = e.getHandle(toks_calo_);
  bool getHits = (hitsCalo.isValid());
  edm::LogVerbatim("HitStudy") << "HOSimHitStudy::Input flag " << hitLab_ << " getHits flag " << getHits;

  std::vector<PCaloHit> zdcHits;
  if (getHits) {
    zdcHits.insert(zdcHits.end(), hitsCalo->begin(), hitsCalo->end());
    unsigned int isiz = zdcHits.size();
    edm::LogVerbatim("HitStudy") << "ZDCSimHitStudy:: Hit buffer for " << hitLab_ << " has " << isiz << " hits";
  }
  analyzeHits(zdcHits);
}

void ZDCSimHitStudy::analyzeHits(const std::vector<PCaloHit> &zdcHits) {
  //initialize
  double etot(0), etotT(0);
  int nHit = zdcHits.size();
  for (int i = 0; i < nHit; i++) {
    double edep = zdcHits[i].energy();
    double time = zdcHits[i].time();
    uint32_t id = zdcHits[i].id();
    int indx = (id & 0xFF);
    etotT += edep;
    if (time < tCut_)
      etot += edep;
    if (verbose_)
      edm::LogVerbatim("HitStudy") << "ZDCSimHitStudy:: Hit " << i << " Section:" << HcalZDCDetId(id).section()
                                   << " zside:" << HcalZDCDetId(id).zside() << " depth:" << HcalZDCDetId(id).depth()
                                   << " channel:" << HcalZDCDetId(id).channel()
                                   << " dense:" << HcalZDCDetId(id).denseIndex() << " edep:" << edep
                                   << " time:" << time;
    time_->Fill(time);
    edep_->Fill(edep);
    indx_->Fill(indx);
  }
  eTot_->Fill(etot);
  eTotT_->Fill(etotT);
  hit_->Fill(nHit);

  if (verbose_)
    edm::LogVerbatim("HitStudy") << "ZDCSimHitStudy::analyzeHits: Hits in ZDC " << nHit << " Energy deposits " << etot
                                 << ":" << etotT;
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZDCSimHitStudy);
