// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <TH1F.h>
#include <TH1I.h>

#define EDM_ML_DEBUG

class XtalDedxAnalysis : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit XtalDedxAnalysis(const edm::ParameterSet &);
  ~XtalDedxAnalysis() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void beginJob() override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override {}

  void analyzeHits(std::vector<PCaloHit> &,
                   edm::Handle<edm::SimTrackContainer> &,
                   edm::Handle<edm::SimVertexContainer> &);

private:
  edm::InputTag caloHitSource_;
  std::string simTkLabel_;

  edm::EDGetTokenT<edm::PCaloHitContainer> tok_calo_;
  edm::EDGetTokenT<edm::SimTrackContainer> tok_tk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tok_vtx_;

  TH1F *meNHit_[4], *meE1T0_[4], *meE9T0_[4], *meE1T1_[4], *meE9T1_[4];
  TH1I *mType_;
};

XtalDedxAnalysis::XtalDedxAnalysis(const edm::ParameterSet &ps) {
  usesResource(TFileService::kSharedResource);
  caloHitSource_ = ps.getParameter<edm::InputTag>("caloHitSource");
  simTkLabel_ = ps.getParameter<std::string>("moduleLabelTk");
  double energyMax = ps.getParameter<double>("energyMax");
  edm::LogVerbatim("CherenkovAnalysis") << "XtalDedxAnalysis::Source " << caloHitSource_ << " Track Label "
                                        << simTkLabel_ << " Energy Max " << energyMax;
  // register for data access
  tok_calo_ = consumes<edm::PCaloHitContainer>(caloHitSource_);
  tok_tk_ = consumes<edm::SimTrackContainer>(edm::InputTag(simTkLabel_));
  tok_vtx_ = consumes<edm::SimVertexContainer>(edm::InputTag(simTkLabel_));

  // Book histograms
  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";
  // Histograms for Hits
  std::string types[4] = {"total", "by dE/dx", "by delta-ray", "by bremms"};
  char name[20], title[80];
  for (int i = 0; i < 4; i++) {
    sprintf(name, "Hits%d", i);
    sprintf(title, "Number of hits (%s)", types[i].c_str());
    meNHit_[i] = tfile->make<TH1F>(name, title, 5000, 0., 5000.);
    meNHit_[i]->GetXaxis()->SetTitle(title);
    meNHit_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "E1T0%d", i);
    sprintf(title, "E1 (Loss %s) in GeV", types[i].c_str());
    meE1T0_[i] = tfile->make<TH1F>(name, title, 5000, 0, energyMax);
    meE1T0_[i]->GetXaxis()->SetTitle(title);
    meE1T0_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "E9T0%d", i);
    sprintf(title, "E9 (Loss %s) in GeV", types[i].c_str());
    meE9T0_[i] = tfile->make<TH1F>(name, title, 5000, 0, energyMax);
    meE9T0_[i]->GetXaxis()->SetTitle(title);
    meE9T0_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "E1T1%d", i);
    sprintf(title, "E1 (Loss %s with t < 400 ns) in GeV", types[i].c_str());
    meE1T1_[i] = tfile->make<TH1F>(name, title, 5000, 0, energyMax);
    meE1T1_[i]->GetXaxis()->SetTitle(title);
    meE1T1_[i]->GetYaxis()->SetTitle("Events");
    sprintf(name, "E9T1%d", i);
    sprintf(title, "E9 (Loss %s with t < 400 ns) in GeV", types[i].c_str());
    meE9T1_[i] = tfile->make<TH1F>(name, title, 5000, 0, energyMax);
    meE9T1_[i]->GetXaxis()->SetTitle(title);
    meE9T1_[i]->GetYaxis()->SetTitle("Events");
  }
  sprintf(name, "PDGType");
  sprintf(title, "PDG ID of first level secondary");
  mType_ = tfile->make<TH1I>(name, title, 5000, -2500, 2500);
  mType_->GetXaxis()->SetTitle(title);
  mType_->GetYaxis()->SetTitle("Tracks");
}

void XtalDedxAnalysis::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("caloHitSource", edm::InputTag("g4SimHits", "EcalHitsEB"));
  desc.add<std::string>("moduleLabelTk", "g4SimHits");
  desc.add<double>("energyMax", 2.0);
  descriptions.add("xtalDedxAnalysis", desc);
}

void XtalDedxAnalysis::analyze(const edm::Event &e, const edm::EventSetup &) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CherenkovAnalysis") << "XtalDedxAnalysis::Run = " << e.id().run() << " Event = " << e.id().event();
#endif
  std::vector<PCaloHit> caloHits;
  edm::Handle<edm::PCaloHitContainer> pCaloHits;
  e.getByToken(tok_calo_, pCaloHits);

  std::vector<SimTrack> theSimTracks;
  edm::Handle<edm::SimTrackContainer> simTk;
  e.getByToken(tok_tk_, simTk);

  std::vector<SimVertex> theSimVertex;
  edm::Handle<edm::SimVertexContainer> simVtx;
  e.getByToken(tok_vtx_, simVtx);

  if (pCaloHits.isValid()) {
    caloHits.insert(caloHits.end(), pCaloHits->begin(), pCaloHits->end());
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CherenkovAnalysis") << "XtalDedxAnalysis: Hit buffer " << caloHits.size();
#endif
    analyzeHits(caloHits, simTk, simVtx);
  }
}

void XtalDedxAnalysis::analyzeHits(std::vector<PCaloHit> &hits,
                                   edm::Handle<edm::SimTrackContainer> &SimTk,
                                   edm::Handle<edm::SimVertexContainer> &SimVtx) {
  edm::SimTrackContainer::const_iterator simTrkItr;
  int nHit = hits.size();
  double e10[4], e90[4], e11[4], e91[4], hit[4];
  for (int i = 0; i < 4; i++)
    e10[i] = e90[i] = e11[i] = e91[i] = hit[i] = 0;
  for (int i = 0; i < nHit; i++) {
    double energy = hits[i].energy();
    double time = hits[i].time();
    unsigned int id_ = hits[i].id();
    int trackID = hits[i].geantTrackId();
    int type = 1;
    for (simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++) {
      if (trackID == (int)(simTrkItr->trackId())) {
        int thePID = simTrkItr->type();
        if (thePID == 11)
          type = 2;
        else if (thePID != -13 && thePID != 13)
          type = 3;
        break;
      }
    }
    hit[0]++;
    hit[type]++;
    e90[0] += energy;
    e90[type] += energy;
    if (time < 400) {
      e91[0] += energy;
      e91[type] += energy;
    }
    if (id_ == 22) {
      e10[0] += energy;
      e10[type] += energy;
      if (time < 400) {
        e11[0] += energy;
        e11[type] += energy;
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CherenkovAnalysis") << "XtalDedxAnalysis:Hit[" << i << "] ID " << id_ << " E " << energy
                                          << " time " << time << " track " << trackID << " type " << type;
#endif
  }
  for (int i = 0; i < 4; i++) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CherenkovAnalysis") << "XtalDedxAnalysis:Type(" << i << ") Hit " << hit[i] << " E10 " << e10[i]
                                          << " E11 " << e11[i] << " E90 " << e90[i] << " E91 " << e91[i];
#endif
    meNHit_[i]->Fill(hit[i]);
    meE1T0_[i]->Fill(e10[i]);
    meE9T0_[i]->Fill(e90[i]);
    meE1T1_[i]->Fill(e11[i]);
    meE9T1_[i]->Fill(e91[i]);
  }

  // Type of the secondary (coming directly from a generator level track)
  int nvtx = 0, ntrk = 0, k1 = 0;
  edm::SimVertexContainer::const_iterator simVtxItr;
  for (simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++)
    ntrk++;
  for (simVtxItr = SimVtx->begin(); simVtxItr != SimVtx->end(); simVtxItr++)
    nvtx++;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CherenkovAnalysis") << "XtalDedxAnalysis: " << ntrk << " tracks and " << nvtx << " vertices";
#endif
  for (simTrkItr = SimTk->begin(); simTrkItr != SimTk->end(); simTrkItr++, ++k1) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CherenkovAnalysis") << "Track " << k1 << " PDGId " << simTrkItr->type() << " Vertex ID "
                                          << simTrkItr->vertIndex() << " Generator " << simTrkItr->noGenpart();
#endif
    if (simTrkItr->noGenpart()) {              // This is a secondary
      int vertIndex = simTrkItr->vertIndex();  // Vertex index of origin
      if (vertIndex >= 0 && vertIndex < nvtx) {
        simVtxItr = SimVtx->begin();
        for (int iv = 0; iv < vertIndex; iv++)
          simVtxItr++;
        int parent = simVtxItr->parentIndex(), k2 = 0;
        for (edm::SimTrackContainer::const_iterator trkItr = SimTk->begin(); trkItr != SimTk->end(); trkItr++, ++k2) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("CherenkovAnalysis") << "XtalDedxAnalysis::Track " << k2 << " ID " << trkItr->trackId()
                                                << " (" << parent << ")  Generator " << trkItr->noGenpart();
#endif
          if ((int)trkItr->trackId() == parent) {  // Parent track
            if (!trkItr->noGenpart()) {            // Generator level
#ifdef EDM_ML_DEBUG
              edm::LogVerbatim("CherenkovAnalysis") << "XtalDedxAnalysis::Track found with ID " << simTrkItr->type();
#endif
              mType_->Fill(simTrkItr->type());
            }
            break;
          }
        }
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(XtalDedxAnalysis);
