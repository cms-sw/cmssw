#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <TH1F.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

//#define EDM_ML_DEBUG

class CaloSimHitStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  CaloSimHitStudy(const edm::ParameterSet& ps);
  ~CaloSimHitStudy() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  void analyzeHits(std::vector<PCaloHit>&, int);
  void analyzeHits(const edm::Handle<edm::PSimHitContainer>&, int);
  void analyzeHits(std::vector<PCaloHit>&, std::vector<PCaloHit>&, std::vector<PCaloHit>&);

private:
  const std::string g4Label_;
  const std::vector<std::string> hitLab_;
  const double maxEnergy_, tmax_, eMIP_;
  const bool storeRL_, testNumber_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tokGeom_;
  const edm::EDGetTokenT<edm::HepMCProduct> tok_evt_;
  const std::vector<edm::EDGetTokenT<edm::PCaloHitContainer>> toks_calo_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> toks_track_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> toks_tkHigh_;
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> toks_tkLow_;

  const CaloGeometry* caloGeometry_;
  const HcalGeometry* hcalGeom_;

  const std::vector<std::string> muonLab_ = {"MuonRPCHits", "MuonCSCHits", "MuonDTHits", "MuonGEMHits"};
  const std::vector<std::string> tkHighLab_ = {"TrackerHitsPixelBarrelHighTof",
                                               "TrackerHitsPixelEndcapHighTof",
                                               "TrackerHitsTECHighTof",
                                               "TrackerHitsTIBHighTof",
                                               "TrackerHitsTIDHighTof",
                                               "TrackerHitsTOBHighTof"};
  const std::vector<std::string> tkLowLab_ = {"TrackerHitsPixelBarrelLowTof",
                                              "TrackerHitsPixelEndcapLowTof",
                                              "TrackerHitsTECLowTof",
                                              "TrackerHitsTIBLowTof",
                                              "TrackerHitsTIDLowTof",
                                              "TrackerHitsTOBLowTof"};

  static constexpr int nCalo_ = 9;
  static constexpr int nTrack_ = 16;
  TH1F *hit_[nCalo_], *time_[nCalo_], *edepEM_[nCalo_], *edepHad_[nCalo_];
  TH1F *edep_[nCalo_], *etot_[nCalo_], *etotg_[nCalo_], *timeAll_[nCalo_];
  TH1F *edepC_[nCalo_], *edepT_[nCalo_], *eta_[nCalo_], *phi_[nCalo_];
  TH1F *hitMu, *hitHigh, *hitLow, *eneInc_, *etaInc_, *phiInc_, *ptInc_;
  TH1F *hitTk_[nTrack_], *edepTk_[nTrack_], *tofTk_[nTrack_];
};

CaloSimHitStudy::CaloSimHitStudy(const edm::ParameterSet& ps)
    : g4Label_(ps.getUntrackedParameter<std::string>("ModuleLabel")),
      hitLab_(ps.getUntrackedParameter<std::vector<std::string>>("CaloCollection")),
      maxEnergy_(ps.getUntrackedParameter<double>("MaxEnergy", 200.0)),
      tmax_(ps.getUntrackedParameter<double>("TimeCut", 100.0)),
      eMIP_(ps.getUntrackedParameter<double>("MIPCut", 0.70)),
      storeRL_(ps.getUntrackedParameter<bool>("StoreRL", false)),
      testNumber_(ps.getUntrackedParameter<bool>("TestNumbering", true)),
      tokGeom_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      tok_evt_(consumes<edm::HepMCProduct>(
          edm::InputTag(ps.getUntrackedParameter<std::string>("SourceLabel", "VtxSmeared")))),
      toks_calo_{edm::vector_transform(hitLab_, [this](const std::string& name) {
        return consumes<edm::PCaloHitContainer>(edm::InputTag{g4Label_, name});
      })} {
  usesResource(TFileService::kSharedResource);

  for (unsigned i = 0; i != muonLab_.size(); i++)
    toks_track_.emplace_back(consumes<edm::PSimHitContainer>(edm::InputTag(g4Label_, muonLab_[i])));
  for (unsigned i = 0; i != tkHighLab_.size(); i++)
    toks_tkHigh_.emplace_back(consumes<edm::PSimHitContainer>(edm::InputTag(g4Label_, tkHighLab_[i])));
  for (unsigned i = 0; i != tkLowLab_.size(); i++)
    toks_tkLow_.emplace_back(consumes<edm::PSimHitContainer>(edm::InputTag(g4Label_, tkLowLab_[i])));

  edm::LogVerbatim("HitStudy") << "Module Label: " << g4Label_ << "   Hits: " << hitLab_[0] << ", " << hitLab_[1]
                               << ", " << hitLab_[2] << ", " << hitLab_[3] << "   MaxEnergy: " << maxEnergy_
                               << "  Tmax: " << tmax_ << "  MIP Cut: " << eMIP_;

  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  char name[20], title[120];
  sprintf(title, "Incident PT (GeV)");
  ptInc_ = tfile->make<TH1F>("PtInc", title, 1000, 0., maxEnergy_);
  ptInc_->GetXaxis()->SetTitle(title);
  ptInc_->GetYaxis()->SetTitle("Events");
  sprintf(title, "Incident Energy (GeV)");
  eneInc_ = tfile->make<TH1F>("EneInc", title, 1000, 0., maxEnergy_);
  eneInc_->GetXaxis()->SetTitle(title);
  eneInc_->GetYaxis()->SetTitle("Events");
  sprintf(title, "Incident #eta");
  etaInc_ = tfile->make<TH1F>("EtaInc", title, 200, -5., 5.);
  etaInc_->GetXaxis()->SetTitle(title);
  etaInc_->GetYaxis()->SetTitle("Events");
  sprintf(title, "Incident #phi");
  phiInc_ = tfile->make<TH1F>("PhiInc", title, 200, -3.1415926, 3.1415926);
  phiInc_->GetXaxis()->SetTitle(title);
  phiInc_->GetYaxis()->SetTitle("Events");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy: Completed defining histos for incident particle";
#endif
  std::string dets[nCalo_] = {"EB", "EB(APD)", "EB(ATJ)", "EE", "ES", "HB", "HE", "HO", "HF"};
  double nhcMax[nCalo_] = {40000., 2000., 2000., 40000., 10000., 10000., 10000., 2000., 10000.};
  for (int i = 0; i < nCalo_; i++) {
    sprintf(name, "Hit%d", i);
    sprintf(title, "Number of hits in %s", dets[i].c_str());
    hit_[i] = tfile->make<TH1F>(name, title, 1000, 0., nhcMax[i]);
    hit_[i]->GetXaxis()->SetTitle(title);
    hit_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "Time%d", i);
    sprintf(title, "Time of the hit (ns) in %s", dets[i].c_str());
    time_[i] = tfile->make<TH1F>(name, title, 1000, 0., 1000.);
    time_[i]->GetXaxis()->SetTitle(title);
    time_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "TimeAll%d", i);
    sprintf(title, "Hit time (ns) in %s with no check on Track ID", dets[i].c_str());
    timeAll_[i] = tfile->make<TH1F>(name, title, 1000, 0., 1000.);
    timeAll_[i]->GetXaxis()->SetTitle(title);
    timeAll_[i]->GetYaxis()->SetTitle("Hits");
    double ymax = maxEnergy_;
    if (i == 1 || i == 2 || i == 4)
      ymax = 1.0;
    else if (i > 4 && i < 8)
      ymax = 10.0;
    sprintf(name, "Edep%d", i);
    sprintf(title, "Energy deposit (GeV) in %s", dets[i].c_str());
    edep_[i] = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edep_[i]->GetXaxis()->SetTitle(title);
    edep_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "EdepEM%d", i);
    sprintf(title, "Energy deposit (GeV) by EM particles in %s", dets[i].c_str());
    edepEM_[i] = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepEM_[i]->GetXaxis()->SetTitle(title);
    edepEM_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "EdepHad%d", i);
    sprintf(title, "Energy deposit (GeV) by hadrons in %s", dets[i].c_str());
    edepHad_[i] = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepHad_[i]->GetXaxis()->SetTitle(title);
    edepHad_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "Etot%d", i);
    sprintf(title, "Total energy deposit (GeV) in %s", dets[i].c_str());
    etot_[i] = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    etot_[i]->GetXaxis()->SetTitle(title);
    etot_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "EtotG%d", i);
    sprintf(title, "Total energy deposit (GeV) in %s (t < 100 ns)", dets[i].c_str());
    etotg_[i] = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    etotg_[i]->GetXaxis()->SetTitle(title);
    etotg_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "eta%d", i);
    sprintf(title, "#eta of hit point in %s", dets[i].c_str());
    eta_[i] = tfile->make<TH1F>(name, title, 100, -5.0, 5.0);
    eta_[i]->GetXaxis()->SetTitle(title);
    eta_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "phi%d", i);
    sprintf(title, "#phi of hit point in %s", dets[i].c_str());
    phi_[i] = tfile->make<TH1F>(name, title, 100, -M_PI, M_PI);
    phi_[i]->GetXaxis()->SetTitle(title);
    phi_[i]->GetYaxis()->SetTitle("Hits");
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy: Completed defining histos for first level of Calorimeter";
#endif
  std::string detx[9] = {"EB/EE (MIP)",
                         "HB/HE (MIP)",
                         "HB/HE/HO (MIP)",
                         "EB/EE (no MIP)",
                         "HB/HE (no MIP)",
                         "HB/HE/HO (no MIP)",
                         "EB/EE (All)",
                         "HB/HE (All)",
                         "HB/HE/HO (All)"};
  for (int i = 0; i < 9; i++) {
    double ymax = 1.0;
    if (i == 0 || i == 3 || i == 6)
      ymax = maxEnergy_;
    sprintf(name, "EdepCal%d", i);
    sprintf(title, "Energy deposit in %s", detx[i].c_str());
    edepC_[i] = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepC_[i]->GetXaxis()->SetTitle(title);
    edepC_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "EdepCalT%d", i);
    sprintf(title, "Energy deposit (t < %f ns) in %s", tmax_, detx[i].c_str());
    edepT_[i] = tfile->make<TH1F>(name, title, 5000, 0., ymax);
    edepT_[i]->GetXaxis()->SetTitle(title);
    edepT_[i]->GetYaxis()->SetTitle("Events");
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy: Completed defining histos for second level of Calorimeter";
#endif
  hitLow = tfile->make<TH1F>("HitLow", "Number of hits in Track (Low)", 1000, 0, 20000.);
  hitLow->GetXaxis()->SetTitle("Number of hits in Track (Low)");
  hitLow->GetYaxis()->SetTitle("Events");
  hitHigh = tfile->make<TH1F>("HitHigh", "Number of hits in Track (High)", 1000, 0, 5000.);
  hitHigh->GetXaxis()->SetTitle("Number of hits in Track (High)");
  hitHigh->GetYaxis()->SetTitle("Events");
  hitMu = tfile->make<TH1F>("HitMu", "Number of hits in Track (Muon)", 1000, 0, 2000.);
  hitMu->GetXaxis()->SetTitle("Number of hits in Muon");
  hitMu->GetYaxis()->SetTitle("Events");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy: Completed defining histos for general tracking hits";
#endif
  std::string dett[nTrack_] = {"Pixel Barrel (High)",
                               "Pixel Endcap (High)",
                               "TEC (High)",
                               "TIB (High)",
                               "TID (High)",
                               "TOB (High)",
                               "Pixel Barrel (Low)",
                               "Pixel Endcap (Low)",
                               "TEC (Low)",
                               "TIB (Low)",
                               "TID (Low)",
                               "TOB (Low)",
                               "RPC",
                               "CSC",
                               "DT",
                               "GEM"};
  double nhtMax[nTrack_] = {
      500., 500., 1000., 1000., 500., 1000., 5000., 2000., 10000., 5000., 2000., 5000., 500., 1000., 1000., 500.};
  for (int i = 0; i < nTrack_; i++) {
    sprintf(name, "HitTk%d", i);
    sprintf(title, "Number of hits in %s", dett[i].c_str());
    hitTk_[i] = tfile->make<TH1F>(name, title, 1000, 0., nhtMax[i]);
    hitTk_[i]->GetXaxis()->SetTitle(title);
    hitTk_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "TimeTk%d", i);
    sprintf(title, "Time of the hit (ns) in %s", dett[i].c_str());
    tofTk_[i] = tfile->make<TH1F>(name, title, 1000, 0., 200.);
    tofTk_[i]->GetXaxis()->SetTitle(title);
    tofTk_[i]->GetYaxis()->SetTitle("Hits");
    sprintf(name, "EdepTk%d", i);
    sprintf(title, "Energy deposit (GeV) in %s", dett[i].c_str());
    double ymax = (i < 12) ? 0.25 : 0.005;
    edepTk_[i] = tfile->make<TH1F>(name, title, 250, 0., ymax);
    edepTk_[i]->GetXaxis()->SetTitle(title);
    edepTk_[i]->GetYaxis()->SetTitle("Hits");
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy: Completed defining histos for SimHit objects";
#endif
}

void CaloSimHitStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> calonames = {"EcalHitsEB", "EcalHitsEE", "EcalHitsES", "HcalHits"};
  desc.addUntracked<std::string>("SourceLabel", "generatorSmeared");
  desc.addUntracked<std::string>("ModuleLabel", "g4SimHits");
  desc.addUntracked<std::vector<std::string>>("CaloCollection", calonames);
  desc.addUntracked<double>("MaxEnergy", 200.0);
  desc.addUntracked<double>("TimeCut", 100.0);
  desc.addUntracked<double>("MIPCut", 0.70);
  desc.addUntracked<bool>("StoreRL", false);
  desc.addUntracked<bool>("TestNumbering", true);
  descriptions.add("CaloSimHitStudy", desc);
}

void CaloSimHitStudy::analyze(edm::Event const& e, edm::EventSetup const& set) {
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy:Run = " << e.id().run() << " Event = " << e.id().event();

  caloGeometry_ = &set.getData(tokGeom_);
  hcalGeom_ = static_cast<const HcalGeometry*>(caloGeometry_->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));

  const edm::Handle<edm::HepMCProduct> EvtHandle = e.getHandle(tok_evt_);
  const HepMC::GenEvent* myGenEvent = EvtHandle->GetEvent();

  double eInc = 0, etaInc = 0, phiInc = 0;
  HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
  if (p != myGenEvent->particles_end()) {
    eInc = (*p)->momentum().e();
    etaInc = (*p)->momentum().eta();
    phiInc = (*p)->momentum().phi();
  }
  double ptInc = eInc / std::cosh(etaInc);
  ptInc_->Fill(ptInc);
  eneInc_->Fill(eInc);
  etaInc_->Fill(etaInc);
  phiInc_->Fill(phiInc);

  std::vector<PCaloHit> ebHits, eeHits, hcHits;
  for (unsigned int i = 0; i < toks_calo_.size(); i++) {
    bool getHits = false;
    const edm::Handle<edm::PCaloHitContainer>& hitsCalo = e.getHandle(toks_calo_[i]);
    if (hitsCalo.isValid())
      getHits = true;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HitStudy") << "CaloSimHitStudy: Input flags Hits " << getHits;
#endif
    if (getHits) {
      std::vector<PCaloHit> caloHits;
      caloHits.insert(caloHits.end(), hitsCalo->begin(), hitsCalo->end());
      if (i == 0)
        ebHits.insert(ebHits.end(), hitsCalo->begin(), hitsCalo->end());
      else if (i == 1)
        eeHits.insert(eeHits.end(), hitsCalo->begin(), hitsCalo->end());
      else if (i == 3)
        hcHits.insert(hcHits.end(), hitsCalo->begin(), hitsCalo->end());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HitStudy") << "CaloSimHitStudy: Hit buffer " << caloHits.size();
#endif
      analyzeHits(caloHits, i);
    }
  }
  analyzeHits(ebHits, eeHits, hcHits);

  std::vector<PSimHit> muonHits;
  for (unsigned int i = 0; i < toks_track_.size(); i++) {
    const edm::Handle<edm::PSimHitContainer>& hitsTrack = e.getHandle(toks_track_[i]);
    if (hitsTrack.isValid()) {
      muonHits.insert(muonHits.end(), hitsTrack->begin(), hitsTrack->end());
      analyzeHits(hitsTrack, i + 12);
    }
  }
  unsigned int nhmu = muonHits.size();
  hitMu->Fill(double(nhmu));
  std::vector<PSimHit> tkHighHits;
  for (unsigned int i = 0; i < toks_tkHigh_.size(); i++) {
    const edm::Handle<edm::PSimHitContainer>& hitsTrack = e.getHandle(toks_tkHigh_[i]);
    if (hitsTrack.isValid()) {
      tkHighHits.insert(tkHighHits.end(), hitsTrack->begin(), hitsTrack->end());
      analyzeHits(hitsTrack, i);
    }
  }
  unsigned int nhtkh = tkHighHits.size();
  hitHigh->Fill(double(nhtkh));
  std::vector<PSimHit> tkLowHits;
  for (unsigned int i = 0; i < toks_tkLow_.size(); i++) {
    const edm::Handle<edm::PSimHitContainer>& hitsTrack = e.getHandle(toks_tkLow_[i]);
    if (hitsTrack.isValid()) {
      tkLowHits.insert(tkLowHits.end(), hitsTrack->begin(), hitsTrack->end());
      analyzeHits(hitsTrack, i + 6);
    }
  }
  unsigned int nhtkl = tkLowHits.size();
  hitLow->Fill(double(nhtkl));
}

void CaloSimHitStudy::analyzeHits(std::vector<PCaloHit>& hits, int indx) {
  int nHit = hits.size();
  int nHB(0), nHE(0), nHO(0), nHF(0), nEB(0), nEBAPD(0), nEBATJ(0);
#ifdef EDM_ML_DEBUG
  int nBad(0), nEE(0), nES(0);
#endif
  std::map<unsigned int, double> hitMap;
  std::vector<double> etot(nCalo_, 0), etotG(nCalo_, 0);
  for (int i = 0; i < nHit; i++) {
    double edep = hits[i].energy();
    double time = hits[i].time();
    unsigned int id = hits[i].id();
    double edepEM = hits[i].energyEM();
    double edepHad = hits[i].energyHad();
    if (indx == 0) {
      int dep = (hits[i].depth()) & PCaloHit::kEcalDepthIdMask;
      if (dep == 1)
        id |= 0x20000;
      else if (dep == 2)
        id |= 0x40000;
    } else if (indx == 3) {
      if (testNumber_) {
        int subdet(0), ieta(0), iphi(0), z(0), lay(0), depth(0);
        HcalTestNumbering::unpackHcalIndex(id, subdet, z, depth, ieta, iphi, lay);
        HcalDDDRecConstants::HcalID hid1 =
            hcalGeom_->topology().dddConstants()->getHCID(subdet, ieta, iphi, lay, depth);
        int zside = 2 * z - 1;
        HcalDetId hid2(static_cast<HcalSubdetector>(hid1.subdet), (zside * hid1.eta), hid1.phi, hid1.depth);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HitStudy") << "From SIM step subdet:z:depth:eta:phi:lay " << subdet << ":" << z << ":"
                                     << depth << ":" << ieta << ":" << iphi << ":" << lay
                                     << " After getHCID subdet:zside:eta:phi:depth " << hid1.subdet << ":" << zside
                                     << ":" << hid1.eta << ":" << hid1.phi << ":" << hid1.depth << " ID " << hid2;
#endif
        id = hid2.rawId();
      }
    }
    std::map<unsigned int, double>::const_iterator it = hitMap.find(id);
    if (it == hitMap.end()) {
      hitMap.insert(std::pair<unsigned int, double>(id, time));
    }
    int idx = -1;
    if (indx != 3) {
      if (indx == 0) {
        if (storeRL_)
          idx = 0;
        else
          idx = ((hits[i].depth()) & PCaloHit::kEcalDepthIdMask);
      } else
        idx = indx + 2;
      time_[idx]->Fill(time);
      edep_[idx]->Fill(edep);
      edepEM_[idx]->Fill(edepEM);
      edepHad_[idx]->Fill(edepHad);
      if (idx == 0)
        nEB++;
      else if (idx == 1)
        nEBAPD++;
      else if (idx == 2)
        nEBATJ++;
#ifdef EDM_ML_DEBUG
      else if (idx == 3)
        nEE++;
      else if (idx == 4)
        nES++;
      else
        nBad++;
#endif
      if (indx >= 0 && indx < 3) {
        etot[idx] += edep;
        if (time < 100)
          etotG[idx] += edep;
      }
    } else {
      HcalSubdetector subdet = HcalDetId(id).subdet();
      if (subdet == HcalSubdetector::HcalBarrel) {
        idx = indx + 2;
        nHB++;
      } else if (subdet == HcalSubdetector::HcalEndcap) {
        idx = indx + 3;
        nHE++;
      } else if (subdet == HcalSubdetector::HcalOuter) {
        idx = indx + 4;
        nHO++;
      } else if (subdet == HcalSubdetector::HcalForward) {
        idx = indx + 5;
        nHF++;
      }
      if (idx > 0) {
        time_[idx]->Fill(time);
        edep_[idx]->Fill(edep);
        edepEM_[idx]->Fill(edepEM);
        edepHad_[idx]->Fill(edepHad);
        etot[idx] += edep;
        if (time < 100)
          etotG[idx] += edep;
      } else {
#ifdef EDM_ML_DEBUG
        nBad++;
#endif
      }
    }
  }
  if (indx < 3) {
    etot_[indx + 2]->Fill(etot[indx + 2]);
    etotg_[indx + 2]->Fill(etotG[indx + 2]);
    if (indx == 0) {
      etot_[indx]->Fill(etot[indx]);
      etotg_[indx]->Fill(etotG[indx]);
      etot_[indx + 1]->Fill(etot[indx + 1]);
      etotg_[indx + 1]->Fill(etotG[indx + 1]);
      hit_[indx]->Fill(double(nEB));
      hit_[indx + 1]->Fill(double(nEBAPD));
      hit_[indx + 2]->Fill(double(nEBATJ));
    } else {
      hit_[indx + 2]->Fill(double(nHit));
    }
  } else if (indx == 3) {
    hit_[5]->Fill(double(nHB));
    hit_[6]->Fill(double(nHE));
    hit_[7]->Fill(double(nHO));
    hit_[8]->Fill(double(nHF));
    for (int idx = 5; idx < 9; idx++) {
      etot_[idx]->Fill(etot[idx]);
      etotg_[idx]->Fill(etotG[idx]);
    }
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy::analyzeHits: EB " << nEB << ", " << nEBAPD << ", " << nEBATJ
                               << " EE " << nEE << " ES " << nES << " HB " << nHB << " HE " << nHE << " HO " << nHO
                               << " HF " << nHF << " Bad " << nBad << " All " << nHit << " Reduced " << hitMap.size();
#endif
  std::map<unsigned int, double>::const_iterator it = hitMap.begin();
  for (; it != hitMap.end(); it++) {
    double time = it->second;
    GlobalPoint point;
    DetId id(it->first);
    if (indx != 2)
      point = caloGeometry_->getPosition(id);
    int idx = -1;
    if (indx < 3) {
      if (indx == 0) {
        if ((id & 0x20000) != 0)
          idx = indx + 1;
        else if ((id & 0x40000) != 0)
          idx = indx + 1;
        else
          idx = indx;
      } else {
        idx = indx + 2;
      }
      if (idx >= 0 && idx < 5)
        timeAll_[idx]->Fill(time);
    } else if (indx == 3) {
      int idx(-1), subdet(0);
      if (testNumber_) {
        int ieta(0), phi(0), z(0), lay(0), depth(0);
        HcalTestNumbering::unpackHcalIndex(id.rawId(), subdet, z, depth, ieta, phi, lay);
      } else {
        subdet = HcalDetId(id).subdet();
      }
      if (subdet == static_cast<int>(HcalBarrel)) {
        idx = indx + 2;
      } else if (subdet == static_cast<int>(HcalEndcap)) {
        idx = indx + 3;
      } else if (subdet == static_cast<int>(HcalOuter)) {
        idx = indx + 4;
      } else if (subdet == static_cast<int>(HcalForward)) {
        idx = indx + 5;
      }
      if (idx > 0) {
        timeAll_[idx]->Fill(time);
        eta_[idx]->Fill(point.eta());
        phi_[idx]->Fill(point.phi());
      }
    }
  }
}

void CaloSimHitStudy::analyzeHits(const edm::Handle<edm::PSimHitContainer>& hits, int indx) {
  int nHit = 0;
  edm::PSimHitContainer::const_iterator ihit;
  std::string label(" ");
  if (indx >= 0 && indx < 6)
    label = tkHighLab_[indx];
  else if (indx >= 6 && indx < 12)
    label = tkLowLab_[indx - 6];
  else if (indx >= 12 && indx < nTrack_)
    label = muonLab_[indx - 12];
  for (ihit = hits->begin(); ihit != hits->end(); ihit++) {
    edepTk_[indx]->Fill(ihit->energyLoss());
    tofTk_[indx]->Fill(ihit->timeOfFlight());
    nHit++;
  }
  hitTk_[indx]->Fill(float(nHit));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy::analyzeHits: for " << label << " Index " << indx << " # of Hits "
                               << nHit;
#endif
}

void CaloSimHitStudy::analyzeHits(std::vector<PCaloHit>& ebHits,
                                  std::vector<PCaloHit>& eeHits,
                                  std::vector<PCaloHit>& hcHits) {
  double edepEB = 0, edepEBT = 0;
  for (unsigned int i = 0; i < ebHits.size(); i++) {
    double edep = ebHits[i].energy();
    double time = ebHits[i].time();
    if (((ebHits[i].depth()) & PCaloHit::kEcalDepthIdMask) == 0) {
      edepEB += edep;
      if (time < tmax_)
        edepEBT += edep;
    }
  }
  double edepEE = 0, edepEET = 0;
  for (unsigned int i = 0; i < eeHits.size(); i++) {
    double edep = eeHits[i].energy();
    double time = eeHits[i].time();
    edepEE += edep;
    if (time < tmax_)
      edepEET += edep;
  }
  double edepH = 0, edepHT = 0, edepHO = 0, edepHOT = 0;
  for (unsigned int i = 0; i < hcHits.size(); i++) {
    double edep = hcHits[i].energy();
    double time = hcHits[i].time();
    int subdet(0);
    if (testNumber_) {
      int ieta(0), phi(0), z(0), lay(0), depth(0);
      HcalTestNumbering::unpackHcalIndex(hcHits[i].id(), subdet, z, depth, ieta, phi, lay);
    } else {
      HcalDetId id = HcalDetId(hcHits[i].id());
      subdet = id.subdet();
    }
    if (subdet == static_cast<int>(HcalBarrel) || subdet == static_cast<int>(HcalEndcap)) {
      edepH += edep;
      if (time < tmax_)
        edepHT += edep;
    } else if (subdet == static_cast<int>(HcalOuter)) {
      edepHO += edep;
      if (time < tmax_)
        edepHOT += edep;
    }
  }
  double edepE = edepEB + edepEE;
  double edepET = edepEBT + edepEET;
  double edepHC = edepH + edepHO;
  double edepHCT = edepHT + edepHOT;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitStudy") << "CaloSimHitStudy::energy in EB " << edepEB << " (" << edepEBT << ") from "
                               << ebHits.size() << " hits; "
                               << " energy in EE " << edepEE << " (" << edepEET << ") from " << eeHits.size()
                               << " hits; energy in HC " << edepH << ", " << edepHO << " (" << edepHT << ", " << edepHOT
                               << ") from " << hcHits.size() << " hits";
#endif
  edepC_[6]->Fill(edepE);
  edepT_[6]->Fill(edepET);
  edepC_[7]->Fill(edepH);
  edepT_[7]->Fill(edepHT);
  edepC_[8]->Fill(edepHC);
  edepT_[8]->Fill(edepHCT);
  if (edepE < eMIP_) {
    edepC_[0]->Fill(edepE);
    edepC_[1]->Fill(edepH);
    edepC_[2]->Fill(edepHC);
  } else {
    edepC_[3]->Fill(edepE);
    edepC_[4]->Fill(edepH);
    edepC_[5]->Fill(edepHC);
  }
  if (edepET < eMIP_) {
    edepT_[0]->Fill(edepET);
    edepT_[1]->Fill(edepHT);
    edepT_[2]->Fill(edepHCT);
  } else {
    edepT_[3]->Fill(edepET);
    edepT_[4]->Fill(edepHT);
    edepT_[5]->Fill(edepHCT);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloSimHitStudy);
