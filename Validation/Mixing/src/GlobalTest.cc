// -*- C++ -*-
//
// Class:      GlobalTest
//
/**\class TestSuite

   Description: global physics test for Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
//
//

#include "Validation/Mixing/interface/GlobalTest.h"

// system include files
#include <memory>
#include <utility>

#include <string>

#include <fmt/format.h>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "TFile.h"

using namespace edm;

GlobalTest::GlobalTest(const edm::ParameterSet &iConfig)
    : filename_(iConfig.getParameter<std::string>("fileName")),
      minbunch_(iConfig.getParameter<int>("minBunch")),
      maxbunch_(iConfig.getParameter<int>("maxBunch")),
      cfTrackToken_(consumes<CrossingFrame<SimTrack>>(iConfig.getParameter<edm::InputTag>("cfTrackTag"))),
      cfVertexToken_(consumes<CrossingFrame<SimTrack>>(iConfig.getParameter<edm::InputTag>("cfVertexTag"))) {
  std::string ecalsubdetb("");
  std::string ecalsubdete("g4SimHitsEcalHitsEE");
  g4SimHits_EB_Token_ = consumes<CrossingFrame<PCaloHit>>(edm::InputTag("mix", "g4SimHitsEcalHitsEB"));
  g4SimHits_EE_Token_ = consumes<CrossingFrame<PCaloHit>>(edm::InputTag("mix", "g4SimHitsEcalHitsEE"));

  std::cout << "Constructed GlobalTest, filename: " << filename_ << " minbunch: " << minbunch_
            << ", maxbunch: " << maxbunch_ << std::endl;
}

void GlobalTest::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const &) {
  using namespace std;

  ibooker.setCurrentFolder("MixingV/Mixing");
  // book histos

  for (int i = minbunch_; i <= maxbunch_; ++i) {
    int ii = i - minbunch_;
    auto label = fmt::format("NrPileupEvts_{}", i);
    nrPileupsH_[ii] = ibooker.book1D(label, label, 100, 0, 100);
    label = fmt::format("NrVertices_{}", i);
    nrVerticesH_[ii] = ibooker.book1D(label, label, 100, 0, 5000);
    label = fmt::format("NrTracks_{}", i);
    nrTracksH_[ii] = ibooker.book1D(label, label, 100, 0, 10000);
    label = fmt::format("TrackPartId", i);
    trackPartIdH_[ii] = ibooker.book1D(label, label, 100, 0, 100);
    label = fmt::format("CaloEnergyEB", i);
    caloEnergyEBH_[ii] = ibooker.book1D(label, label, 100, 0., 1000.);
    label = fmt::format("CaloEnergyEE", i);
    caloEnergyEEH_[ii] = ibooker.book1D(label, label, 100, 0., 1000.);
  }
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void GlobalTest::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;
  using namespace std;

  // Get input
  edm::Handle<CrossingFrame<SimTrack>> cf_track;
  edm::Handle<CrossingFrame<SimTrack>> cf_vertex;
  edm::Handle<CrossingFrame<PCaloHit>> cf_calohitE;
  edm::Handle<CrossingFrame<PCaloHit>> cf_calohitB;
  std::string ecalsubdetb("g4SimHitsEcalHitsEB");
  std::string ecalsubdete("g4SimHitsEcalHitsEE");
  iEvent.getByToken(cfTrackToken_, cf_track);
  iEvent.getByToken(cfVertexToken_, cf_vertex);
  iEvent.getByToken(g4SimHits_EB_Token_, cf_calohitB);
  iEvent.getByToken(g4SimHits_EE_Token_, cf_calohitE);

  // number of events/bcr ??

  // number of tracks
  for (int i = minbunch_; i <= maxbunch_; ++i) {
    nrTracksH_[i - minbunch_]->Fill(cf_track->getNrPileups(i));
  }

  // number of vertices
  for (int i = minbunch_; i <= maxbunch_; ++i) {
    nrVerticesH_[i - minbunch_]->Fill(cf_vertex->getNrPileups(i));
  }

  // part id for each track
  std::unique_ptr<MixCollection<SimTrack>> coltr(new MixCollection<SimTrack>(cf_track.product()));
  MixCollection<SimTrack>::iterator cfitr;
  for (cfitr = coltr->begin(); cfitr != coltr->end(); cfitr++) {
    trackPartIdH_[cfitr.bunch() - minbunch_]->Fill(cfitr->type());
  }

  // energy sum
  double sumE[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  std::unique_ptr<MixCollection<PCaloHit>> colecalb(new MixCollection<PCaloHit>(cf_calohitB.product()));
  MixCollection<PCaloHit>::iterator cfiecalb;
  for (cfiecalb = colecalb->begin(); cfiecalb != colecalb->end(); cfiecalb++) {
    sumE[cfiecalb.bunch() - minbunch_] += cfiecalb->energy();
    //      if (cfiecal.getTrigger())    tofecalhist_sig->Fill(cfiecal->time());
    //      else    tofecalhist->Fill(cfiecal->time());
  }
  for (int i = minbunch_; i <= maxbunch_; ++i) {
    caloEnergyEBH_[i - minbunch_]->Fill(sumE[i - minbunch_]);
  }
  double sumEE[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  std::unique_ptr<MixCollection<PCaloHit>> colecale(new MixCollection<PCaloHit>(cf_calohitE.product()));
  MixCollection<PCaloHit>::iterator cfiecale;
  for (cfiecale = colecale->begin(); cfiecale != colecale->end(); cfiecale++) {
    sumEE[cfiecale.bunch() - minbunch_] += cfiecale->energy();
    //      if (cfiecal.getTrigger())    tofecalhist_sig->Fill(cfiecal->time());
    //      else    tofecalhist->Fill(cfiecal->time());
  }
  for (int i = minbunch_; i <= maxbunch_; ++i) {
    caloEnergyEEH_[i - minbunch_]->Fill(sumEE[i - minbunch_]);
  }
}
