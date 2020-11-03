// system include files
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include "CLHEP/Geometry/Vector3D.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "TH1D.h"
#include "TH2D.h"

class HcalGeomCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  struct hitsinfo {
    hitsinfo() { phi = eta = energy = time = 0.0; }
    double phi, eta, energy, time;
  };

  explicit HcalGeomCheck(const edm::ParameterSet&);
  ~HcalGeomCheck() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  void analyzeHits(int, const std::string&, const std::vector<PCaloHit>&);

  // ----------member data ---------------------------
  const std::string caloHitSource_;
  const int ietaMin_, ietaMax_, depthMax_;
  const double rmin_, rmax_, zmin_, zmax_;
  const int nbinR_, nbinZ_, verbosity_;
  const HcalDDDRecConstants* hcons_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_HRNDC_;

  //histogram related stuff
  TH2D* h_RZ_;
  std::map<std::pair<int, int>, TH1D*> h_phi_;
  std::map<int, TH1D*> h_eta_;
  std::vector<TH1D*> h_E_, h_T_;
};

HcalGeomCheck::HcalGeomCheck(const edm::ParameterSet& iConfig)
    :

      caloHitSource_(iConfig.getParameter<std::string>("caloHitSource")),
      ietaMin_(iConfig.getUntrackedParameter<int>("ietaMin", -41)),
      ietaMax_(iConfig.getUntrackedParameter<int>("ietaMax", 41)),
      depthMax_(iConfig.getUntrackedParameter<int>("depthMax", 7)),
      rmin_(iConfig.getUntrackedParameter<double>("rMin", 0.0)),
      rmax_(iConfig.getUntrackedParameter<double>("rMax", 5500.0)),
      zmin_(iConfig.getUntrackedParameter<double>("zMin", -12500.0)),
      zmax_(iConfig.getUntrackedParameter<double>("zMax", 12500.0)),
      nbinR_(iConfig.getUntrackedParameter<int>("nBinR", 550)),
      nbinZ_(iConfig.getUntrackedParameter<int>("nBinZ", 2500)),
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)) {
  usesResource(TFileService::kSharedResource);
  tok_hits_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", caloHitSource_));
  tok_HRNDC_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>();
}

void HcalGeomCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("caloHitSource", "HcalHits");
  desc.addUntracked<int>("ietaMin", -41);
  desc.addUntracked<int>("ietaMax", 41);
  desc.addUntracked<int>("depthMax", 7);
  desc.addUntracked<double>("rMin", 0.0);
  desc.addUntracked<double>("rMax", 5500.0);
  desc.addUntracked<double>("zMin", -12500.0);
  desc.addUntracked<double>("zMax", 12500.0);
  desc.addUntracked<int>("nBinR", 550);
  desc.addUntracked<int>("nBinZ", 250);
  desc.addUntracked<int>("verbosity", 0);
  descriptions.add("hcalGeomCheck", desc);
}

void HcalGeomCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Now the hits
  edm::Handle<edm::PCaloHitContainer> theCaloHitContainer;
  iEvent.getByToken(tok_hits_, theCaloHitContainer);
  if (theCaloHitContainer.isValid()) {
    if (verbosity_ > 0)
      edm::LogVerbatim("HcalValidation") << " PcalohitItr = " << theCaloHitContainer->size();

    //Merge hits for the same DetID
    std::map<HcalDetId, hitsinfo> map_hits;
    unsigned int nused(0);
    for (auto const& hit : *(theCaloHitContainer.product())) {
      unsigned int id = hit.id();
      HcalDetId detId = HcalHitRelabeller::relabel(id, hcons_);
      double energy = hit.energy();
      double time = hit.time();
      int subdet = detId.subdet();
      int ieta = detId.ieta();
      int iphi = detId.iphi();
      int depth = detId.depth();
      std::pair<double, double> etaphi = hcons_->getEtaPhi(subdet, ieta, iphi);
      double rz = hcons_->getRZ(subdet, ieta, depth);
      if (verbosity_ > 2)
        edm::LogVerbatim("HcalValidation") << "i/p " << subdet << ":" << ieta << ":" << iphi << ":" << depth << " o/p "
                                           << etaphi.first << ":" << etaphi.second << ":" << rz;
      HepGeom::Point3D<float> gcoord = HepGeom::Point3D<float>(rz * cos(etaphi.second) / cosh(etaphi.first),
                                                               rz * sin(etaphi.second) / cosh(etaphi.first),
                                                               rz * tanh(etaphi.first));
      nused++;
      double tof = (gcoord.mag() * CLHEP::mm) / CLHEP::c_light;
      if (verbosity_ > 1)
        edm::LogVerbatim("HcalValidation")
            << "Detector " << subdet << " ieta = " << ieta << " iphi = " << iphi << " depth = " << depth
            << " positon = " << gcoord << " energy = " << energy << " time = " << time << ":" << tof;
      time -= tof;
      if (time < 0)
        time = 0;
      hitsinfo hinfo;
      if (map_hits.count(detId) != 0) {
        hinfo = map_hits[detId];
      } else {
        hinfo.phi = gcoord.getPhi();
        hinfo.eta = gcoord.getEta();
        hinfo.time = time;
      }
      hinfo.energy += energy;
      map_hits[detId] = hinfo;

      h_RZ_->Fill(gcoord.z(), gcoord.rho());
    }

    //Fill in histograms
    for (auto const& hit : map_hits) {
      hitsinfo hinfo = hit.second;
      int subdet = hit.first.subdet();
      if (subdet > 0 && subdet < static_cast<int>(h_E_.size())) {
        int depth = hit.first.depth();
        int ieta = hit.first.ieta();
        int iphi = hit.first.iphi();
        if (verbosity_ > 1)
          edm::LogVerbatim("HGCalValidation")
              << " ----------------------   eta = " << ieta << ":" << hinfo.eta << " phi = " << iphi << ":" << hinfo.phi
              << " depth = " << depth << " E = " << hinfo.energy << " T = " << hinfo.time;
        h_E_[subdet]->Fill(hinfo.energy);
        h_T_[subdet]->Fill(hinfo.time);
        auto itr1 = h_phi_.find(std::pair<int, int>(ieta, depth));
        if (itr1 != h_phi_.end())
          itr1->second->Fill(iphi);
        auto itr2 = h_eta_.find(depth);
        if (itr2 != h_eta_.end())
          itr2->second->Fill(ieta);
      }
    }
  } else if (verbosity_ > 0) {
    edm::LogVerbatim("HcalValidation") << "PCaloHitContainer does not "
                                       << "exist for HCAL";
  }
}

// ------------ method called when starting to processes a run  ------------
void HcalGeomCheck::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  const auto& pHRNDC = iSetup.getData(tok_HRNDC_);
  hcons_ = &pHRNDC;
  if (verbosity_ > 0)
    edm::LogVerbatim("HcalValidation") << "Obtain HcalDDDRecConstants from Event Setup";
}

void HcalGeomCheck::beginJob() {
  if (verbosity_ > 2)
    edm::LogVerbatim("HcalValidation") << "HcalGeomCheck:: Enter beginJob";
  edm::Service<TFileService> fs;
  h_RZ_ = fs->make<TH2D>("RZ", "R vs Z", nbinZ_, zmin_, zmax_, nbinR_, rmin_, rmax_);
  if (verbosity_ > 2)
    edm::LogVerbatim("HcalValidation") << "HcalGeomCheck: booked scatterplot RZ";
  char name[20], title[100];
  for (int depth = 1; depth < depthMax_; ++depth) {
    for (int ieta = ietaMin_; ieta <= ietaMax_; ++ieta) {
      sprintf(name, "phi%d%d", ieta, depth);
      sprintf(title, "i#phi (i#eta = %d, depth = %d)", ieta, depth);
      h_phi_[std::pair<int, int>(ieta, depth)] = fs->make<TH1D>(name, title, 400, -20, 380);
      if (verbosity_ > 2)
        edm::LogVerbatim("HcalValidation") << "HcalGeomCheck: books " << title;
    }
    sprintf(name, "eta%d", depth);
    sprintf(title, "i#eta (depth = %d)", depth);
    h_eta_[depth] = fs->make<TH1D>(name, title, 100, -50, 50);
    if (verbosity_ > 2)
      edm::LogVerbatim("HcalValidation") << "HcalGeomCheck: books " << title;
  }

  std::vector<std::string> dets = {"HCAL", "HB", "HE", "HF", "HO"};
  for (unsigned int ih = 0; ih < dets.size(); ++ih) {
    sprintf(name, "E_%s", dets[ih].c_str());
    sprintf(title, "Energy deposit in %s (MeV)", dets[ih].c_str());
    h_E_.emplace_back(fs->make<TH1D>(name, title, 1000, 0.0, 1.0));
    if (verbosity_ > 2)
      edm::LogVerbatim("HcalValidation") << "HcalGeomCheck: books " << title;
    sprintf(name, "T_%s", dets[ih].c_str());
    sprintf(title, "Time of hit in %s (ns)", dets[ih].c_str());
    h_T_.emplace_back(fs->make<TH1D>(name, title, 1000, 0.0, 200.0));
    if (verbosity_ > 2)
      edm::LogVerbatim("HcalValidation") << "HcalGeomCheck: books " << title;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HcalGeomCheck);
