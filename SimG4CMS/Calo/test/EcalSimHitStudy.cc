#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <TH1F.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class EcalSimHitStudy: public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:

  EcalSimHitStudy(const edm::ParameterSet& ps);
  ~EcalSimHitStudy() {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:

  virtual void beginJob() override;
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyzeHits  (std::vector<PCaloHit> &, int);

private:

  static const int    ndets_ = 2;
  std::string         g4Label_, hitLab_[ndets_];
  edm::EDGetTokenT<edm::HepMCProduct>      tok_evt_;
  edm::EDGetTokenT<edm::PCaloHitContainer> toks_calo_[2];
  double              maxEnergy_, tmax_, w0_;
  const CaloGeometry *geometry_;
  TH1F               *ptInc_, *etaInc_, *phiInc_, *eneInc_;
  TH1F               *hit_[ndets_],     *time_[ndets_],    *timeAll_[ndets_];
  TH1F               *edepEM_[ndets_],  *edepHad_[ndets_], *edep_[ndets_];
  TH1F               *etot_[ndets_],    *etotg_[ndets_],   *edepAll_[ndets_];
  TH1F               *r1by9_[ndets_],   *r1by25_[ndets_],  *r9by25_[ndets_];
  TH1F               *sEtaEta_[ndets_], *sEtaPhi_[ndets_], *sPhiPhi_[ndets_];
};


EcalSimHitStudy::EcalSimHitStudy(const edm::ParameterSet& ps) {

  usesResource(TFileService::kSharedResource);

  g4Label_  = ps.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits");
  hitLab_[0]= ps.getUntrackedParameter<std::string>("EBCollection","EcalHitsEB");
  hitLab_[1]= ps.getUntrackedParameter<std::string>("EECollection","EcalHitsEE");
  tok_evt_  = consumes<edm::HepMCProduct>(edm::InputTag(ps.getUntrackedParameter<std::string>("SourceLabel","VtxSmeared")));
  maxEnergy_= ps.getUntrackedParameter<double>("MaxEnergy", 200.0);
  tmax_     = ps.getUntrackedParameter<double>("TimeCut", 100.0);
  w0_       = ps.getUntrackedParameter<double>("W0", 4.7);

  for (int i=0; i<ndets_; ++i)
    toks_calo_[i] = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_,hitLab_[i]));

  edm::LogVerbatim("HitStudy") << "Module Label: " << g4Label_ << "   Hits: "
			       << hitLab_[0] << ", " << hitLab_[1] 
			       << "   MaxEnergy: " << maxEnergy_ << "  Tmax: "
			       << tmax_;
}

void  EcalSimHitStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("ModuleLabel","g4SimHits");
  desc.addUntracked<std::string>("EBCollection","EcalHitsEB");
  desc.addUntracked<std::string>("EECollection","EcalHitsEE");
  desc.addUntracked<std::string>("SourceLabel","VtxSmeared");
  desc.addUntracked<double>("MaxEnergy", 200.0);
  desc.addUntracked<double>("TimeCut", 100.0);
  descriptions.add("EcalSimHitStudy",desc);
}

void EcalSimHitStudy::beginJob() {

  edm::Service<TFileService> tfile;
  if ( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  char  name[20], title[120];
  sprintf (title, "Incident PT (GeV)");
  ptInc_ = tfile->make<TH1F>("PtInc", title, 1000, 0., maxEnergy_);
  ptInc_->GetXaxis()-> SetTitle(title);
  ptInc_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Incident Energy (GeV)");
  eneInc_ = tfile->make<TH1F>("EneInc", title, 1000, 0., maxEnergy_);
  eneInc_->GetXaxis()-> SetTitle(title);
  eneInc_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Incident #eta");
  etaInc_ = tfile->make<TH1F>("EtaInc", title, 200, -5., 5.);
  etaInc_->GetXaxis()->SetTitle(title);
  etaInc_->GetYaxis()->SetTitle("Events");
  sprintf (title, "Incident #phi");
  phiInc_ = tfile->make<TH1F>("PhiInc", title, 200, -3.1415926, 3.1415926);
  phiInc_->GetXaxis()->SetTitle(title);
  phiInc_->GetYaxis()->SetTitle("Events");
  std::string dets[ndets_] = {"EB", "EE"};
  for (int i=0; i<ndets_; i++) {
    sprintf (name, "Hit%d", i);
    sprintf (title, "Number of hits in %s", dets[i].c_str());
    hit_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 20000.);
    hit_[i]->GetXaxis()->SetTitle(title); 
    hit_[i]->GetYaxis()->SetTitle("Events");
    hit_[i]->Sumw2();
    sprintf (name, "Time%d", i);
    sprintf (title, "Time of the hit (ns) in %s", dets[i].c_str());
    time_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 1000.);
    time_[i]->GetXaxis()->SetTitle(title); 
    time_[i]->GetYaxis()->SetTitle("Hits");
    time_[i]->Sumw2();
    sprintf (name, "TimeAll%d", i);
    sprintf (title, "Hit time (ns) in %s (for first hit in the cell)", dets[i].c_str());
    timeAll_[i]  = tfile->make<TH1F>(name, title, 1000, 0., 1000.);
    timeAll_[i]->GetXaxis()->SetTitle(title); 
    timeAll_[i]->GetYaxis()->SetTitle("Hits");
    timeAll_[i]->Sumw2();
    sprintf (name, "Edep%d", i);
    sprintf (title, "Energy deposit (GeV) in %s", dets[i].c_str());
    edep_[i]  = tfile->make<TH1F>(name, title, 5000, 0., maxEnergy_);
    edep_[i]->GetXaxis()->SetTitle(title); 
    edep_[i]->GetYaxis()->SetTitle("Hits");
    edep_[i]->Sumw2();
    sprintf (name, "EdepAll%d", i);
    sprintf (title, "Total Energy deposit in the cell (GeV) in %s", dets[i].c_str());
    edepAll_[i]  = tfile->make<TH1F>(name, title, 5000, 0., maxEnergy_);
    edepAll_[i]->GetXaxis()->SetTitle(title); 
    edepAll_[i]->GetYaxis()->SetTitle("Hits");
    edepAll_[i]->Sumw2();
    sprintf (name, "EdepEM%d", i);
    sprintf (title, "Energy deposit (GeV) by EM particles in %s", dets[i].c_str());
    edepEM_[i]  = tfile->make<TH1F>(name, title, 5000, 0., maxEnergy_);
    edepEM_[i]->GetXaxis()->SetTitle(title); 
    edepEM_[i]->GetYaxis()->SetTitle("Hits");
    edepEM_[i]->Sumw2();
    sprintf (name, "EdepHad%d", i);
    sprintf (title, "Energy deposit (GeV) by hadrons in %s", dets[i].c_str());
    edepHad_[i]  = tfile->make<TH1F>(name, title, 5000, 0., maxEnergy_);
    edepHad_[i]->GetXaxis()->SetTitle(title); 
    edepHad_[i]->GetYaxis()->SetTitle("Hits");
    edepHad_[i]->Sumw2();
    sprintf (name, "Etot%d", i);
    sprintf (title, "Total energy deposit (GeV) in %s", dets[i].c_str());
    etot_[i]  = tfile->make<TH1F>(name, title, 5000, 0., maxEnergy_);
    etot_[i]->GetXaxis()->SetTitle(title); 
    etot_[i]->GetYaxis()->SetTitle("Events");
    etot_[i]->Sumw2();
    sprintf (name, "EtotG%d", i);
    sprintf (title, "Total energy deposit (GeV) in %s (t < 100 ns)", dets[i].c_str());
    etotg_[i]  = tfile->make<TH1F>(name, title, 5000, 0., maxEnergy_);
    etotg_[i]->GetXaxis()->SetTitle(title); 
    etotg_[i]->GetYaxis()->SetTitle("Events");
    etotg_[i]->Sumw2();
    sprintf (name, "r1by9%d", i);
    sprintf (title, "E1/E9 in %s", dets[i].c_str());
    r1by9_[i]  = tfile->make<TH1F>(name, title, 100, 0.0, 1.0);
    r1by9_[i]->GetXaxis()->SetTitle(title); 
    r1by9_[i]->GetYaxis()->SetTitle("Events");
    r1by9_[i]->Sumw2();
    sprintf (name, "r1by25%d", i);
    sprintf (title, "E1/E25 in %s", dets[i].c_str());
    r1by25_[i]  = tfile->make<TH1F>(name, title, 100, 0.0, 1.0);
    r1by25_[i]->GetXaxis()->SetTitle(title); 
    r1by25_[i]->GetYaxis()->SetTitle("Events");
    r1by25_[i]->Sumw2();
    sprintf (name, "r9by25%d", i);
    sprintf (title, "E9/E25 in %s", dets[i].c_str());
    r9by25_[i]  = tfile->make<TH1F>(name, title, 100, 0.0, 1.0);
    r9by25_[i]->GetXaxis()->SetTitle(title); 
    r9by25_[i]->GetYaxis()->SetTitle("Events");
    r9by25_[i]->Sumw2();
    double ymax = (i == 0) ? 0.0005 : 0.005;
    sprintf (name, "sEtaEta%d", i);
    sprintf (title, "Cov(#eta,#eta) inn %s", dets[i].c_str());
    sEtaEta_[i]  = tfile->make<TH1F>(name, title, 1000, 0.0, ymax);
    sEtaEta_[i]->GetXaxis()->SetTitle(title); 
    sEtaEta_[i]->GetYaxis()->SetTitle("Events");
    sEtaEta_[i]->Sumw2();
    sprintf (name, "sEtaPhi%d", i);
    sprintf (title, "Cov(#eta,#phi) inn %s", dets[i].c_str());
    sEtaPhi_[i]  = tfile->make<TH1F>(name, title, 1000, 0.0, ymax);
    sEtaPhi_[i]->GetXaxis()->SetTitle(title); 
    sEtaPhi_[i]->GetYaxis()->SetTitle("Events");
    sEtaPhi_[i]->Sumw2();
    ymax = (i == 0) ? 0.001 : 0.01;
    sprintf (name, "sPhiPhi%d", i);
    sprintf (title, "Cov(#phi,#phi) inn %s", dets[i].c_str());
    sPhiPhi_[i]  = tfile->make<TH1F>(name, title, 1000, 0.0, ymax);
    sPhiPhi_[i]->GetXaxis()->SetTitle(title); 
    sPhiPhi_[i]->GetYaxis()->SetTitle("Events");
    sPhiPhi_[i]->Sumw2();
  }
}

void EcalSimHitStudy::analyze(const edm::Event& e, const edm::EventSetup& iS) {

  edm::LogVerbatim("HitStudy") << "Run = " << e.id().run() << " Event = " 
			       << e.id().event();

  // get handles to calogeometry
  edm::ESHandle<CaloGeometry> pG;
  iS.get<CaloGeometryRecord>().get(pG);
  geometry_ = pG.product();

  edm::Handle<edm::HepMCProduct > EvtHandle;
  e.getByToken(tok_evt_, EvtHandle);
  const  HepMC::GenEvent* myGenEvent = EvtHandle->GetEvent();

  double eInc=0, etaInc=0, phiInc=0;
  HepMC::GenEvent::particle_const_iterator p=myGenEvent->particles_begin();
  if (p != myGenEvent->particles_end()) {
    eInc   = (*p)->momentum().e();
    etaInc = (*p)->momentum().eta();
    phiInc = (*p)->momentum().phi();
  }
  double ptInc = eInc/std::cosh(etaInc);
  ptInc_->Fill(ptInc);
  eneInc_->Fill(eInc);
  etaInc_->Fill(etaInc);
  phiInc_->Fill(phiInc);

  int type(-1);
  if      (std::abs(etaInc) < 1.46)                           type = 0;
  else if (std::abs(etaInc) > 1.49 && std::abs(etaInc) < 3.0) type = 1;
  if (type >= 0) {
    bool getHits(false);
    unsigned int nhits(0);
    edm::Handle<edm::PCaloHitContainer> hitsCalo;
    e.getByToken(toks_calo_[type],hitsCalo); 
    if (hitsCalo.isValid()) {getHits = true; nhits = hitsCalo->size();}
    edm::LogVerbatim("HitStudy") << "EcalSimHitStudy: Input flags Hits " 
				 << getHits << " with " << nhits << " hits";
    if (getHits) {
      std::vector<PCaloHit> caloHits;
      caloHits.insert(caloHits.end(),hitsCalo->begin(),hitsCalo->end());
      if (caloHits.size() > 0) analyzeHits (caloHits, type);
    }
  }
}

void EcalSimHitStudy::analyzeHits (std::vector<PCaloHit>& hits, int indx) {

  unsigned int nEC(0);
  std::map<unsigned int,std::pair<double,double> > hitMap;
  double       etot(0), etotG(0);
  for (auto hit : hits) {
    double edep      = hit.energy();
    double time      = hit.time();
    unsigned int id_ = hit.id();
    double edepEM    = hit.energyEM();
    double edepHad   = hit.energyHad();
    if (indx == 0) {
      if ((hit.depth()==1) || (hit.depth()==2)) continue;
    }
    if (time <= tmax_) {
      auto it = hitMap.find(id_);
      if (it == hitMap.end()) {
	hitMap[id_] = std::pair<double,double>(time,edep);
      } else {
	(it->second).second += edep;
      }
      etotG += edep; 
      ++nEC;
    }
    time_[indx]->Fill(time);
    edep_[indx]->Fill(edep);
    edepEM_[indx]->Fill(edepEM);
    edepHad_[indx]->Fill(edepHad);
    etot += edep;
  }

  edm::LogVerbatim("HitStudy") << "EcalSimHitStudy::analyzeHits: Index " << indx
			       << " Hits " << hits.size() << ":" << nEC << ":" 
			       << hitMap.size() << " ETotal " << etot << ":" 
			       << etotG;

  etot_[indx]->Fill(etot);
  etotg_[indx]->Fill(etotG);
  hit_[indx]->Fill(double(nEC));
  double       edepM(0);
  unsigned int idM(0);
  for (auto it : hitMap) {
    timeAll_[indx]->Fill((it.second).first);
    edepAll_[indx]->Fill((it.second).second);
    if ((it.second).second > edepM) {
      edepM = (it.second).second;
      idM   = it.first;
    }
  }

  math::XYZVector              meanPosition(0.0, 0.0, 0.0);
  std::vector<math::XYZVector> position;
  std::vector<double>          energy;
  double                       e9(0), e25(0);
  for (auto it : hitMap) {
    DetId id(it.first);
    int   deta(99), dphi(99), dz(0);
    if (indx == 0) {
      deta = std::abs(EBDetId(id).ietaAbs()-EBDetId(idM).ietaAbs());
      dphi = std::abs(EBDetId(id).iphi()-EBDetId(idM).iphi());
      if (dphi > 180) dphi = std::abs(dphi-360);
      dz   = std::abs(EBDetId(id).zside()-EBDetId(idM).zside());
    } else {
      deta = std::abs(EEDetId(id).ix()-EEDetId(idM).ix());
      dphi = std::abs(EEDetId(id).iy()-EEDetId(idM).iy());
      dz   = std::abs(EEDetId(id).zside()-EEDetId(idM).zside());
    }
    if (deta <= 1 && dphi <= 1 && dz < 1) e9 += (it.second).second;
    if (deta <= 2 && dphi <= 2 && dz < 1) {
      e25 += (it.second).second;
      GlobalPoint gpos = geometry_->getGeometry(id)->getPosition();
      math::XYZVector pos(gpos.x(),gpos.y(),gpos.z());
      meanPosition += (it.second).second * pos;
      position.push_back(pos); energy.push_back((it.second).second);
    }
  }
  double r1by9  = (e9 > 0)  ? (edepM/e9)  : -1;
  double r1by25 = (e25 > 0) ? (edepM/e25) : -1;
  double r9by25 = (e25 > 0) ? (e9/e25)    : -1;

  meanPosition /= e25;
  double denom(0), numEtaEta(0), numEtaPhi(0), numPhiPhi(0);
  for (unsigned int k=0; k<position.size(); ++k) {
    double dEta = position[k].eta() - meanPosition.eta();
    double dPhi = position[k].phi() - meanPosition.phi();
    if (dPhi > +M_PI) { dPhi = 2*M_PI - dPhi; }
    if (dPhi < -M_PI) { dPhi = 2*M_PI + dPhi; }

    double w   = std::max(0.0,(w0_+std::log(energy[k]/e25)));
    denom     += w;
    numEtaEta += std::abs(w * dEta * dEta);
    numEtaPhi += std::abs(w * dEta * dPhi);
    numPhiPhi += std::abs(w * dPhi * dPhi);
    edm::LogVerbatim("HitStudy") << "[" << k << "] dEta " << dEta << " dPhi "
				 << dPhi << " Wt " << energy[k]/e25 << ":" 
				 << std::log(energy[k]/e25) << ":" << w;
  }
  double sEtaEta = (denom > 0) ? (numEtaEta / denom) : -1.0;
  double sEtaPhi = (denom > 0) ? (numEtaPhi / denom) : -1.0;
  double sPhiPhi = (denom > 0) ? (numPhiPhi / denom) : -1.0;

  edm::LogVerbatim("HitStudy") << "EcalSimHitStudy::Ratios " << r1by9 << " : "
			       << r1by25 << " : " << r9by25 << " Covariances "
			       << sEtaEta << " : " << sEtaPhi << " : "
			       << sPhiPhi;

  r1by9_[indx]->Fill(r1by9);     r1by25_[indx]->Fill(r1by25);
  r9by25_[indx]->Fill(r9by25);   sEtaEta_[indx]->Fill(sEtaEta);
  sEtaPhi_[indx]->Fill(sEtaPhi); sPhiPhi_[indx]->Fill(sPhiPhi);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalSimHitStudy);

