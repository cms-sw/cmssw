#include "Validation/RecoB/plugins/BDHadronTrackMonitoringHarvester.h"

using namespace edm;
using namespace std;
using namespace RecoBTag;

// intialize category map
// std::map<unsigned int, std::string>
// BDHadronTrackMonitoringAnalyzer::TrkHistCat(map_start_values,
// map_start_values
// + map_start_values_size);

// typedef std::map<unsigned int, std::string>::iterator it_type;

BDHadronTrackMonitoringHarvester::BDHadronTrackMonitoringHarvester(const edm::ParameterSet &pSet) {}

BDHadronTrackMonitoringHarvester::~BDHadronTrackMonitoringHarvester() {}

void BDHadronTrackMonitoringHarvester::beginJob() {}

void BDHadronTrackMonitoringHarvester::dqmEndJob(DQMStore::IBooker &ibook, DQMStore::IGetter &iget) {
  // ***********************
  //
  // Book all new histograms.
  //
  // ***********************
  RecoBTag::setTDRStyle();
  ibook.setCurrentFolder("BDHadronTracks/JetContent");

  // b jets
  // absolute average number of tracks
  nTrk_absolute_bjet = ibook.book1D("nTrk_absolute_bjet", "absolute average number of tracks in b jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_absolute_bjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_absolute_bjet->setAxisRange(0, 5, 2);
  nTrk_absolute_bjet->setAxisTitle("average number of tracks", 2);

  // relative (in percent) average number of tracks
  nTrk_relative_bjet = ibook.book1D("nTrk_relative_bjet", "relative average number of tracks in b jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_relative_bjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_relative_bjet->setAxisRange(0, 1, 2);
  nTrk_relative_bjet->setAxisTitle("average fraction of tracks", 2);

  // standard deviation of number of tracks
  nTrk_std_bjet = ibook.book1D("nTrk_std_bjet", "RMS of number of tracks in b jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_std_bjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_std_bjet->setAxisRange(0, 3, 2);
  nTrk_std_bjet->setAxisTitle("RMS of number of tracks", 2);

  // c jets
  nTrk_absolute_cjet = ibook.book1D("nTrk_absolute_cjet", "absolute average number of tracks in c jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_absolute_cjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_absolute_cjet->setAxisRange(0, 5, 2);
  nTrk_absolute_cjet->setAxisTitle("average number of tracks", 2);

  nTrk_relative_cjet = ibook.book1D("nTrk_relative_cjet", "relative average number of tracks in c jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_relative_cjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_relative_cjet->setAxisRange(0, 1, 2);
  nTrk_relative_cjet->setAxisTitle("average fraction of tracks", 2);

  nTrk_std_cjet = ibook.book1D("nTrk_std_cjet", "RMS of number of tracks in c jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_std_cjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_std_cjet->setAxisRange(0, 3, 2);
  nTrk_std_cjet->setAxisTitle("RMS of number of tracks", 2);

  // udsg jets
  nTrk_absolute_dusgjet =
      ibook.book1D("nTrk_absolute_dusgjet", "absolute average number of tracks in dusg jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_absolute_dusgjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_absolute_dusgjet->setAxisRange(0, 5, 2);
  nTrk_absolute_dusgjet->setAxisTitle("average number of tracks", 2);

  nTrk_relative_dusgjet =
      ibook.book1D("nTrk_relative_dusgjet", "relative average number of tracks in dusg jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_relative_dusgjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_relative_dusgjet->setAxisRange(0, 1, 2);
  nTrk_relative_dusgjet->setAxisTitle("average fraction of tracks", 2);

  nTrk_std_dusgjet = ibook.book1D("nTrk_std_dusgjet", "RMS of number of tracks in dusg jets", 6, -0.5, 5.5);
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_std_dusgjet->setBinLabel(i + 1, BDHadronTrackMonitoringAnalyzer::TrkHistCat[i], 1);
  }
  nTrk_std_dusgjet->setAxisRange(0, 3, 2);
  nTrk_std_dusgjet->setAxisTitle("RMS of number of tracks", 2);

  // ***********************
  //
  // get all the old histograms
  //
  // ***********************

  // b jets
  MonitorElement *nTrk_bjet[6];
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_bjet[i] = iget.get("BDHadronTracks/JetContent/nTrk_bjet_" + BDHadronTrackMonitoringAnalyzer::TrkHistCat[i]);
  }
  MonitorElement *nTrkAll_bjet = iget.get("BDHadronTracks/JetContent/nTrkAll_bjet");

  // c jets
  MonitorElement *nTrk_cjet[6];
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_cjet[i] = iget.get("BDHadronTracks/JetContent/nTrk_cjet_" + BDHadronTrackMonitoringAnalyzer::TrkHistCat[i]);
  }
  MonitorElement *nTrkAll_cjet = iget.get("BDHadronTracks/JetContent/nTrkAll_cjet");

  // dusg jets
  MonitorElement *nTrk_dusgjet[6];
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    nTrk_dusgjet[i] =
        iget.get("BDHadronTracks/JetContent/nTrk_dusgjet_" + BDHadronTrackMonitoringAnalyzer::TrkHistCat[i]);
  }
  MonitorElement *nTrkAll_dusgjet = iget.get("BDHadronTracks/JetContent/nTrkAll_dusgjet");

  // ***********************
  //
  // Calculate contents of new histograms
  //
  // ***********************

  // b jets
  float mean_bjets[6];
  float std_bjets[6];
  float meanAll_bjets;
  meanAll_bjets = std::max(0.01, nTrkAll_bjet->getMean(1));
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    mean_bjets[i] = nTrk_bjet[i]->getMean(1);  // mean number of tracks per category
    std_bjets[i] = nTrk_bjet[i]->getRMS(1);
    nTrk_absolute_bjet->setBinContent(i + 1, mean_bjets[i]);
    nTrk_relative_bjet->setBinContent(i + 1, mean_bjets[i] / meanAll_bjets);
    nTrk_std_bjet->setBinContent(i + 1, std_bjets[i]);
  }

  // c jets
  float mean_cjets[6];
  float std_cjets[6];
  float meanAll_cjets;
  meanAll_cjets = std::max(0.01, nTrkAll_cjet->getMean(1));
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    mean_cjets[i] = nTrk_cjet[i]->getMean(1);  // mean number of tracks per category
    std_cjets[i] = nTrk_cjet[i]->getRMS(1);
    nTrk_absolute_cjet->setBinContent(i + 1, mean_cjets[i]);
    nTrk_relative_cjet->setBinContent(i + 1, mean_cjets[i] / meanAll_cjets);
    nTrk_std_cjet->setBinContent(i + 1, std_cjets[i]);
  }

  // dusg jets
  float mean_dusgjets[6];
  float std_dusgjets[6];
  float meanAll_dusgjets;
  meanAll_dusgjets = std::max(0.01, nTrkAll_dusgjet->getMean(1));
  for (unsigned int i = 0; i < BDHadronTrackMonitoringAnalyzer::TrkHistCat.size(); i++) {
    mean_dusgjets[i] = nTrk_dusgjet[i]->getMean(1);  // mean number of tracks per category
    std_dusgjets[i] = nTrk_dusgjet[i]->getRMS(1);
    nTrk_absolute_dusgjet->setBinContent(i + 1, mean_dusgjets[i]);
    nTrk_relative_dusgjet->setBinContent(i + 1, mean_dusgjets[i] / meanAll_dusgjets);
    nTrk_std_dusgjet->setBinContent(i + 1, std_dusgjets[i]);
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(BDHadronTrackMonitoringHarvester);
