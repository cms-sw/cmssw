// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

// Root objects
#include "TH1.h"
#include "TH2.h"

class HGCalSiliconValidation : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCalSiliconValidation(const edm::ParameterSet& ps);
  ~HGCalSiliconValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  const std::string g4Label_, nameDetector_, hgcalHits_;
  const edm::InputTag hgcalDigis_;
  const int iSample_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcalgeom_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;
  const edm::EDGetTokenT<HGCalDigiCollection> tok_digi_;

  TH1D *hsimE1_, *hsimE2_, *hsimTm_;
  TH1D *hsimLn_, *hdigEn_, *hdigLn_;
  TH2D *hsimOc_, *hsi2Oc_, *hdigOc_, *hdi2Oc_;
};

HGCalSiliconValidation::HGCalSiliconValidation(const edm::ParameterSet& ps)
    : g4Label_(ps.getUntrackedParameter<std::string>("ModuleLabel", "g4SimHits")),
      nameDetector_(ps.getUntrackedParameter<std::string>("detectorName", "HGCalEESensitive")),
      hgcalHits_((ps.getUntrackedParameter<std::string>("HitCollection", "HGCHitsEE"))),
      hgcalDigis_(ps.getUntrackedParameter<edm::InputTag>("DigiCollection")),
      iSample_(ps.getUntrackedParameter<int>("Sample", 5)),
      tok_hgcalgeom_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameDetector_})),
      tok_hits_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hgcalHits_))),
      tok_digi_(consumes<HGCalDigiCollection>(hgcalDigis_)) {
  usesResource(TFileService::kSharedResource);

  edm::LogVerbatim("HGCalValidation") << "HGCalSiliconValidation::Input for SimHit:"
                                      << edm::InputTag(g4Label_, hgcalHits_) << "  Digits:" << hgcalDigis_
                                      << "  Sample: " << iSample_;
}

void HGCalSiliconValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("ModuleLabel", "g4SimHits");
  desc.addUntracked<std::string>("detectorName", "HGCalEESensitive");
  desc.addUntracked<std::string>("HitCollection", "HGCHitsEE");
  desc.addUntracked<edm::InputTag>("DigiCollection", edm::InputTag("simHGCalUnsuppressedDigis", "EE"));
  desc.addUntracked<int>("Sample", 5);
  descriptions.add("hgcalSiliconAnalysisEE", desc);
}

void HGCalSiliconValidation::beginRun(edm::Run const&, edm::EventSetup const& es) {
  edm::LogVerbatim("HGCalValidation") << "HGCalSiliconValidation::Booking the Histograms";

  //Histograms for Sim Hits
  edm::Service<TFileService> fs;
  hsimE1_ = fs->make<TH1D>("SimHitEn1", "Sim Hit Energy", 1000, 0.0, 1.0);
  hsimE2_ = fs->make<TH1D>("SimHitEn2", "Sim Hit Energy", 1000, 0.0, 1.0);
  hsimTm_ = fs->make<TH1D>("SimHitTime", "Sim Hit Time", 1000, 0.0, 500.0);
  hsimLn_ = fs->make<TH1D>("SimHitLong", "Sim Hit Long. Profile", 60, 0.0, 30.0);
  hsimOc_ = fs->make<TH2D>("SimHitOccup", "Sim Hit Occupnacy", 300, 0.0, 300.0, 60, 0.0, 30.0);
  hsi2Oc_ = fs->make<TH2D>("SimHitOccu2", "Sim Hit Occupnacy", 300, 300.0, 600.0, 300, 0.0, 300.0);
  //Histograms for Digis
  hdigEn_ = fs->make<TH1D>("DigiEnergy", "Digi ADC Sample", 1000, 0.0, 1000.0);
  hdigLn_ = fs->make<TH1D>("DigiLong", "Digi Long. Profile", 60, 0.0, 30.0);
  hdigOc_ = fs->make<TH2D>("DigiOccup", "Digi Occupnacy", 300, 0.0, 300.0, 60, 0.0, 30.0);
  hdi2Oc_ = fs->make<TH2D>("DigiOccu2", "Digi Occupnacy", 300, 300.0, 600.0, 300, 0.0, 300.0);
}

void HGCalSiliconValidation::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::LogVerbatim("HGCalValidation") << "HGCalSiliconValidation:Run = " << e.id().run()
                                      << " Event = " << e.id().event();

  const edm::ESHandle<HGCalGeometry>& geom = iSetup.getHandle(tok_hgcalgeom_);
  if (!geom.isValid()) {
    edm::LogWarning("HGCalValidation") << "Cannot get valid HGCalGeometry Object for " << nameDetector_;
  } else {
    const HGCalGeometry* geom0 = geom.product();

    //SimHits
    const edm::Handle<edm::PCaloHitContainer>& hitsCalo = e.getHandle(tok_hits_);
    edm::LogVerbatim("HGCalValidation") << "HGCalSiliconValidation.: PCaloHitContainer obtained with flag "
                                        << hitsCalo.isValid();
    if (hitsCalo.isValid()) {
      edm::LogVerbatim("HGCalValidation") << "HGCalSiliconValidation: PCaloHit buffer " << hitsCalo->size();
      unsigned i(0);
      std::map<unsigned int, double> map_try;
      for (edm::PCaloHitContainer::const_iterator it = hitsCalo->begin(); it != hitsCalo->end(); ++it) {
        double energy = it->energy();
        double time = it->time();
        unsigned int id = it->id();
        GlobalPoint pos = geom0->getPosition(DetId(id));
        double r = pos.perp();
        double z = std::abs(pos.z());
        int lay = HGCSiliconDetId(id).layer();
        hsimE1_->Fill(energy);
        hsimTm_->Fill(time, energy);
        hsimOc_->Fill(r, lay, energy);
        hsi2Oc_->Fill(z, r, energy);
        hsimLn_->Fill(lay, energy);
        double ensum = (map_try.count(id) != 0) ? map_try[id] : 0;
        ensum += energy;
        map_try[id] = ensum;
        ++i;
        edm::LogVerbatim("HGCalValidation") << "HGCalHit[" << i << "] ID " << std::hex << " " << id << std::dec << " "
                                            << HGCSiliconDetId(id) << " E " << energy << " time " << time;
      }
      for (std::map<unsigned int, double>::iterator itr = map_try.begin(); itr != map_try.end(); ++itr) {
        hsimE2_->Fill((*itr).second);
      }
    }

    //Digits
    unsigned int kount(0);
    const edm::Handle<HGCalDigiCollection>& digicoll = e.getHandle(tok_digi_);
    edm::LogVerbatim("HGCalValidation") << "HGCalSiliconValidation.: HGCalDigiCollection obtained with flag "
                                        << digicoll.isValid();
    if (digicoll.isValid()) {
      edm::LogVerbatim("HGCalValidation") << "HGCalSiliconValidation: HGCalDigi buffer " << digicoll->size();
      for (HGCalDigiCollection::const_iterator it = digicoll->begin(); it != digicoll->end(); ++it) {
        HGCalDataFrame df(*it);
        double energy = df[iSample_].data();
        HGCSiliconDetId cell(df.id());
        GlobalPoint pos = geom0->getPosition(cell);
        double r = pos.perp();
        double z = std::abs(pos.z());
        int depth = cell.layer();
        hdigEn_->Fill(energy);
        hdigLn_->Fill(depth);
        hdigOc_->Fill(r, depth);
        hdi2Oc_->Fill(z, r);
        ++kount;
        edm::LogVerbatim("HGCalValidation") << "HGCalDigit[" << kount << "] ID " << cell << " E " << energy;
      }
    }
  }
}

DEFINE_FWK_MODULE(HGCalSiliconValidation);
