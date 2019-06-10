#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include <TH1F.h>
#include <TH2F.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

//#define EDM_ML_DEBUG

class CaloSimHitAnalysis : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  CaloSimHitAnalysis(const edm::ParameterSet& ps);
  ~CaloSimHitAnalysis() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  void analyzeHits(std::vector<PCaloHit>&, int);

private:
  const std::string              g4Label_;
  const std::vector<std::string> hitLab_;
  const std::vector<double>      timeSliceUnit_;
  const double                   maxEnergy_, maxTime_, tMax_, tScale_;
  const bool                     testNumber_;
  std::vector<edm::EDGetTokenT<edm::PCaloHitContainer> > toks_calo_;

  const CaloGeometry* caloGeometry_;
  const HcalGeometry* hcalGeom_;

  static constexpr int nCalo_ = 6;
  TH1F *h_hit_[nCalo_], *h_time_[nCalo_], *h_edep_[nCalo_];
  TH1F *h_edepEM_[nCalo_], *h_edepHad_[nCalo_], *h_rr_[nCalo_], *h_zz_[nCalo_];
  TH1F *h_eta_[nCalo_], *h_phi_[nCalo_], *h_etot_[nCalo_], *h_etotg_[nCalo_];
  TH2F *h_rz_, *h_etaphi_;
};

CaloSimHitAnalysis::CaloSimHitAnalysis(const edm::ParameterSet& ps) : 
  g4Label_(ps.getUntrackedParameter<std::string>("moduleLabel","g4SimHits")),
  hitLab_(ps.getParameter<std::vector<std::string> >("hitCollection")),
  timeSliceUnit_(ps.getParameter<std::vector<double> >("timeSliceUnit")),
  maxEnergy_(ps.getUntrackedParameter<double>("maxEnergy", 200.0)),
  maxTime_(ps.getUntrackedParameter<double>("maxTime", 1000.0)),
  tMax_(ps.getUntrackedParameter<double>("timeCut", 100.0)),
  tScale_(ps.getUntrackedParameter<double>("timeScale", 1.0)),
  testNumber_(ps.getUntrackedParameter<bool>("TestNumbering", false)) {

  usesResource(TFileService::kSharedResource);

  // register for data access
  for (unsigned int i = 0; i < hitLab_.size(); i++)
    toks_calo_.emplace_back(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hitLab_[i])));

  edm::LogVerbatim("HitStudy") 
    << "Module Label: " << g4Label_ << "   Hits|timeSliceUnit:";
  for (unsigned int i = 0; i < hitLab_.size(); i++)
    edm::LogVerbatim("HitStudy") 
      << "[" << i << "] "  << hitLab_[i] << " " << timeSliceUnit_[i];
  edm::LogVerbatim("HitStudy")  
    << "maxEnergy: " << maxEnergy_ << " maxTime: " << maxTime_ << " tMax: " 
    << tMax_;

  edm::Service<TFileService> tfile;
  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  char name[20], title[120];
  std::string dets[nCalo_] = {"EB", "EE", "HB", "HE", "HO", "HF"};
  for (int i = 0; i < nCalo_; i++) {
    sprintf(name, "Hit%d", i);
    sprintf(title, "Number of hits in %s", dets[i].c_str());
    h_hit_[i] = tfile->make<TH1F>(name, title, 100, 0., 20000.);
    h_hit_[i]->GetXaxis()->SetTitle(title);
    h_hit_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "Time%d", i);
    sprintf(title, "Time of the hit (ns) in %s", dets[i].c_str());
    h_time_[i] = tfile->make<TH1F>(name, title, 100, 0., 200.);
    h_time_[i]->GetXaxis()->SetTitle(title);
    h_time_[i]->GetYaxis()->SetTitle("Hits");
    double ymax = 0.1;
    if (i > 1) ymax = 0.01;
    sprintf(name, "Edep%d", i);
    sprintf(title, "Energy deposit (GeV) in %s", dets[i].c_str());
    h_edep_[i] = tfile->make<TH1F>(name, title, 100, 0., ymax);
    h_edep_[i]->GetXaxis()->SetTitle(title);
    h_edep_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "EdepEM%d", i);
    sprintf(title, "Energy deposit (GeV) by EM particles in %s", dets[i].c_str());
    h_edepEM_[i] = tfile->make<TH1F>(name, title, 100, 0., ymax);
    h_edepEM_[i]->GetXaxis()->SetTitle(title);
    h_edepEM_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "EdepHad%d", i);
    sprintf(title, "Energy deposit (GeV) by hadrons in %s", dets[i].c_str());
    h_edepHad_[i] = tfile->make<TH1F>(name, title, 100, 0., ymax);
    h_edepHad_[i]->GetXaxis()->SetTitle(title);
    h_edepHad_[i]->GetYaxis()->SetTitle("Hits");
    ymax = (i > 1) ? 1.0 : maxEnergy_;
    sprintf(name, "Etot%d", i);
    sprintf(title, "Total energy deposit (GeV) in %s", dets[i].c_str());
    h_etot_[i] = tfile->make<TH1F>(name, title, 100, 0., ymax);
    h_etot_[i]->GetXaxis()->SetTitle(title);
    h_etot_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "EtotG%d", i);
    sprintf(title, "Total energy deposit (GeV) in %s (t < 100 ns)", dets[i].c_str());
    h_etotg_[i] = tfile->make<TH1F>(name, title, 100, 0., ymax);
    h_etotg_[i]->GetXaxis()->SetTitle(title);
    h_etotg_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "rr%d", i);
    sprintf(title, "R of hit point (cm) in %s", dets[i].c_str());
    h_rr_[i] = tfile->make<TH1F>(name, title, 100, 0., 500.);
    h_rr_[i]->GetXaxis()->SetTitle(title);
    h_rr_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "zz%d", i);
    sprintf(title, "z of hit point (cm) in %s", dets[i].c_str());
    h_zz_[i] = tfile->make<TH1F>(name, title, 240, -1200., 1200.);
    h_zz_[i]->GetXaxis()->SetTitle(title);
    h_zz_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "eta%d", i);
    sprintf(title, "#eta of hit point in %s", dets[i].c_str());
    h_eta_[i] = tfile->make<TH1F>(name, title, 100, -5.0, 5.0);
    h_eta_[i]->GetXaxis()->SetTitle(title);
    h_eta_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "phi%d", i);
    sprintf(title, "#phi of hit point in %s", dets[i].c_str());
    h_phi_[i] = tfile->make<TH1F>(name, title, 100, -M_PI, M_PI);
    h_phi_[i]->GetXaxis()->SetTitle(title);
    h_phi_[i]->GetYaxis()->SetTitle("Hits");
  }
  sprintf(title, "R vs Z of hit point");
  h_rz_ = tfile->make<TH2F>("rz", title, 120, 0., 1200., 100, 0., 500.);
  h_rz_->GetXaxis()->SetTitle("z (cm)");
  h_rz_->GetYaxis()->SetTitle("R (cm)");
  sprintf(title, "#phi vs #eta of hit point");
  h_etaphi_ = tfile->make<TH2F>("etaphi", title, 100, 0., 5., 100, 0., M_PI);
  h_etaphi_->GetXaxis()->SetTitle("#eta");
  h_etaphi_->GetYaxis()->SetTitle("#phi");
}

void CaloSimHitAnalysis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> labels = {"EcalHitsEB1","EcalHitsEE1","HcalHits1"};
  std::vector<double>      times  = {1,1,1};
  desc.addUntracked<std::string>("moduleLabel", "g4SimHits");
  desc.add<std::vector<std::string> >("hitCollection", labels);
  desc.add<std::vector<double> >("timeSliceUnit", times);
  desc.addUntracked<double>("maxEnergy", 200.0);
  desc.addUntracked<double>("maxTime", 1000.0);
  desc.addUntracked<double>("timeCut", 100.0);
  desc.addUntracked<double>("timeScale", 1.0);
  desc.addUntracked<bool>("testNumbering", false);
  descriptions.add("caloSimHitAnalysis", desc);
}

void CaloSimHitAnalysis::analyze(edm::Event const& e, edm::EventSetup const& set) {
  edm::LogVerbatim("HitStudy") 
    << "CaloSimHitAnalysis:Run = " << e.id().run() << " Event = " 
    << e.id().event();

  edm::ESHandle<CaloGeometry> calo_handle;
  set.get<CaloGeometryRecord>().get(calo_handle);
  caloGeometry_ = calo_handle.product();
  hcalGeom_ = static_cast<const HcalGeometry*>(caloGeometry_->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));

  for (unsigned int i = 0; i < toks_calo_.size(); i++) {
    edm::Handle<edm::PCaloHitContainer> hitsCalo;
    e.getByToken(toks_calo_[i], hitsCalo);
    bool getHits = (hitsCalo.isValid());
    edm::LogVerbatim("HitStudy") 
      << "CaloSimHitAnalysis: Input flags Hits[" << i << "]: " << getHits;

    if (getHits) {
      std::vector<PCaloHit> caloHits;
      caloHits.insert(caloHits.end(), hitsCalo->begin(), hitsCalo->end());
      edm::LogVerbatim("HitStudy") 
	<< "CaloSimHitAnalysis: Hit buffer [" << i << "] " << caloHits.size();
      analyzeHits(caloHits, i);
    }
  }
}

void CaloSimHitAnalysis::analyzeHits(std::vector<PCaloHit>& hits, int indx) {
  int nHit = hits.size();
  int nHB = 0, nHE = 0, nHO = 0, nHF = 0, nEB = 0, nEE = 0, nBad = 0, iHit=0;
  std::map<CaloHitID, std::pair<double,double> > hitMap;
  double etot[nCalo_], etotG[nCalo_];
  for (unsigned int k=0; k<nCalo_; ++k) {etot[k] = etotG[k] = 0;}
  for (const auto& hit : hits) {
    double edep = hit.energy();
    double time = tScale_*hit.time();
    uint32_t id = hit.id();
    int    itra = hit.geantTrackId();
    double edepEM = hit.energyEM();
    double edepHad = hit.energyHad();
    int    idx(-1);
    if (indx != 2) {
      idx = indx;
      if (indx == 0) ++nEB;
      else           ++nEE;
    } else {
      int subdet(0);
      if (testNumber_) {
        int ieta(0), phi(0), z(0), lay(0), depth(0);
        HcalTestNumbering::unpackHcalIndex(id,subdet,z,depth,ieta,phi,lay);
	id  = HcalDetId(static_cast<HcalSubdetector>(subdet),z*ieta,phi,depth).rawId();
      } else {
        subdet = HcalDetId(id).subdet();
      }
      if (subdet == static_cast<int>(HcalBarrel)) {
        idx = indx;
        nHB++;
      } else if (subdet == static_cast<int>(HcalEndcap)) {
        idx = indx + 1;
        nHE++;
      } else if (subdet == static_cast<int>(HcalOuter)) {
        idx = indx + 2;
        nHO++;
      } else if (subdet == static_cast<int>(HcalForward)) {
        idx = indx + 3;
        nHF++;
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HitStudy") 
      << "Hit[" << iHit << ":" << nHit << ":" << idx << "] E " << edep << ":"
      << edepEM << ":" << edepHad << " T " << time << " itra " << itra << " ID "
      << std::hex << id << std::dec;
#endif
    ++iHit;
    if (idx >= 0) {
      CaloHitID  hid(id,time,itra,0,timeSliceUnit_[indx]);
      auto itr = hitMap.find(hid);
      if (itr == hitMap.end()) {
	hitMap[hid] = std::make_pair(time, edep);
      } else {
	((itr->second).second) += edep;
      }
      h_edepEM_[idx]->Fill(edepEM);
      h_edepHad_[idx]->Fill(edepHad);
    } else {
      ++nBad;
    }
  }

  //Now make plots
  for (auto itr = hitMap.begin(); itr != hitMap.end(); ++itr) {
    int idx = -1;
    GlobalPoint point;
    DetId id((itr->first).unitID());
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HitStudy") 
      << "Index " << indx << " Geom " << caloGeometry_ << ":" << hcalGeom_ 
      << "  " << std::hex << id.rawId() << std::dec;
#endif
    if (indx != 2) {
      idx = indx;
      point = caloGeometry_->getPosition(id);
    } else {
      int subdet = id.subdetId();
      if (subdet == static_cast<int>(HcalBarrel)) {
        idx = indx;
      } else if (subdet == static_cast<int>(HcalEndcap)) {
        idx = indx + 1;
      } else if (subdet == static_cast<int>(HcalOuter)) {
        idx = indx + 2;
      } else if (subdet == static_cast<int>(HcalForward)) {
        idx = indx + 3;
      }
      point = hcalGeom_->getPosition(id);
    }
    double edep = (itr->second).second;
    double time = (itr->second).first;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HitStudy") 
      << "Index " << idx << ":" << nCalo_ << " Point " << point << " E " 
      << edep << " T " << time;
#endif
    if (idx >= 0) {
      h_time_[idx]->Fill(time);
      h_edep_[idx]->Fill(edep);
      h_rr_[idx]->Fill(point.perp());
      h_zz_[idx]->Fill(point.z());
      h_eta_[idx]->Fill(point.eta());
      h_phi_[idx]->Fill(point.phi());
      h_rz_->Fill(std::abs(point.z()),point.perp());
      h_etaphi_->Fill(std::abs(point.eta()),std::abs(point.phi()));
      etot[idx] += edep;
      if (time < tMax_) etotG[idx] += edep;
    }
  }

  if (indx < 2) {
    h_etot_[indx]->Fill(etot[indx]);
    h_etotg_[indx]->Fill(etotG[indx]);
    if (indx == 0) h_hit_[indx]->Fill(double(nEB));
    else           h_hit_[indx]->Fill(double(nEE));
  } else {
    h_hit_[2]->Fill(double(nHB));
    h_hit_[3]->Fill(double(nHE));
    h_hit_[4]->Fill(double(nHO));
    h_hit_[5]->Fill(double(nHF));
    for (int idx = 2; idx < nCalo_; idx++) {
      h_etot_[idx]->Fill(etot[idx]);
      h_etotg_[idx]->Fill(etotG[idx]);
    }
  }

  edm::LogVerbatim("HitStudy") 
    << "CaloSimHitAnalysis::analyzeHits: EB " << nEB << " EE " << nEE << " HB "
    << nHB << " HE " << nHE << " HO " << nHO << " HF " << nHF << " Bad " 
    << nBad << " All " << nHit << " Reduced " << hitMap.size();
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloSimHitAnalysis);
