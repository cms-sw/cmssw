#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>

// class TrackAssociator;
class TrackAssociatorByHits;
class TrackerHitAssociator;

namespace reco {
  class TrackToTrackingParticleAssociator;
}

class testTrackAssociator : public edm::one::EDAnalyzer<> {
public:
  testTrackAssociator(const edm::ParameterSet &conf);
  ~testTrackAssociator() override = default;
  void beginJob() override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  reco::TrackToTrackingParticleAssociator const *associatorByChi2;
  reco::TrackToTrackingParticleAssociator const *associatorByHits;
  edm::InputTag tracksTag, tpTag, simtracksTag, simvtxTag;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenMF_;
};

using namespace reco;
using namespace std;
using namespace edm;

testTrackAssociator::testTrackAssociator(edm::ParameterSet const &conf) {
  tracksTag = conf.getParameter<edm::InputTag>("tracksTag");
  tpTag = conf.getParameter<edm::InputTag>("tpTag");
  simtracksTag = conf.getParameter<edm::InputTag>("simtracksTag");
  simvtxTag = conf.getParameter<edm::InputTag>("simvtxTag");
  tokenMF_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
}

void testTrackAssociator::analyze(const edm::Event &event, const edm::EventSetup &setup) {
  //const auto &theMF = setup.getHandle(tokenMF_);
  edm::Handle<reco::TrackToTrackingParticleAssociator> theChiAssociator;
  event.getByLabel("trackAssociatorByChi2", theChiAssociator);
  associatorByChi2 = theChiAssociator.product();
  edm::Handle<reco::TrackToTrackingParticleAssociator> theHitsAssociator;
  event.getByLabel("trackAssociatorByHits", theHitsAssociator);
  associatorByHits = theHitsAssociator.product();

  Handle<View<Track>> trackCollectionH;
  event.getByLabel(tracksTag, trackCollectionH);
  const View<Track> &tC = *(trackCollectionH.product());

  Handle<SimTrackContainer> simTrackCollection;
  event.getByLabel(simtracksTag, simTrackCollection);
  const SimTrackContainer &simTC = *(simTrackCollection.product());

  Handle<SimVertexContainer> simVertexCollection;
  event.getByLabel(simvtxTag, simVertexCollection);

  edm::Handle<TrackingParticleCollection> TPCollectionH;
  event.getByLabel(tpTag, TPCollectionH);

  cout << "\nEvent ID = " << event.id() << endl;

  // RECOTOSIM
  cout << "                      ****************** Reco To Sim "
          "****************** "
       << endl;
  cout << "-- Associator by hits --" << endl;
  reco::RecoToSimCollection p = associatorByHits->associateRecoToSim(trackCollectionH, TPCollectionH);
  for (View<Track>::size_type i = 0; i < tC.size(); ++i) {
    RefToBase<Track> track(trackCollectionH, i);
    try {
      std::vector<std::pair<TrackingParticleRef, double>> tp = p[track];
      cout << "Reco Track pT: " << setw(6) << track->pt() << " matched to " << tp.size() << " MC Tracks" << std::endl;
      for (std::vector<std::pair<TrackingParticleRef, double>>::const_iterator it = tp.begin(); it != tp.end(); ++it) {
        TrackingParticleRef tpr = it->first;
        double assocChi2 = it->second;
        cout << "\t\tMCTrack " << setw(2) << tpr.index() << " pT: " << setw(6) << tpr->pt() << " NShared: " << assocChi2
             << endl;
      }
    } catch (Exception const &) {
      cout << "->   Track pT: " << setprecision(2) << setw(6) << track->pt() << " matched to 0  MC Tracks" << endl;
    }
  }
  cout << "-- Associator by chi2 --" << endl;
  p = associatorByChi2->associateRecoToSim(trackCollectionH, TPCollectionH);
  for (View<Track>::size_type i = 0; i < tC.size(); ++i) {
    RefToBase<Track> track(trackCollectionH, i);
    try {
      std::vector<std::pair<TrackingParticleRef, double>> tp = p[track];
      cout << "Reco Track pT: " << setw(6) << track->pt() << " matched to " << tp.size() << " MC Tracks" << std::endl;
      for (std::vector<std::pair<TrackingParticleRef, double>>::const_iterator it = tp.begin(); it != tp.end(); ++it) {
        TrackingParticleRef tpr = it->first;
        double assocChi2 = it->second;
        cout << "\t\tMCTrack " << setw(2) << tpr.index() << " pT: " << setw(6) << tpr->pt() << " chi2: " << assocChi2
             << endl;
      }
    } catch (Exception const &) {
      cout << "->   Track pT: " << setprecision(2) << setw(6) << track->pt() << " matched to 0  MC Tracks" << endl;
    }
  }
  // SIMTORECO
  cout << "                      ****************** Sim To Reco "
          "****************** "
       << endl;
  cout << "-- Associator by hits --" << endl;
  reco::SimToRecoCollection q = associatorByHits->associateSimToReco(trackCollectionH, TPCollectionH);
  for (SimTrackContainer::size_type i = 0; i < simTC.size(); ++i) {
    TrackingParticleRef tp(TPCollectionH, i);
    try {
      std::vector<std::pair<RefToBase<Track>, double>> trackV = q[tp];
      cout << "Sim Track " << setw(2) << tp.index() << " pT: " << setw(6) << tp->pt() << " matched to " << trackV.size()
           << " reco::Tracks" << std::endl;
      for (std::vector<std::pair<RefToBase<Track>, double>>::const_iterator it = trackV.begin(); it != trackV.end();
           ++it) {
        RefToBase<Track> tr = it->first;
        double assocChi2 = it->second;
        cout << "\t\treco::Track pT: " << setw(6) << tr->pt() << " NShared: " << assocChi2 << endl;
      }
    } catch (Exception const &) {
      cout << "->   TrackingParticle " << setw(2) << tp.index() << " pT: " << setprecision(2) << setw(6) << tp->pt()
           << " matched to 0  reco::Tracks" << endl;
    }
  }
  cout << "-- Associator by chi2 --" << endl;
  q = associatorByChi2->associateSimToReco(trackCollectionH, TPCollectionH);
  for (SimTrackContainer::size_type i = 0; i < simTC.size(); ++i) {
    TrackingParticleRef tp(TPCollectionH, i);
    try {
      std::vector<std::pair<RefToBase<Track>, double>> trackV = q[tp];
      cout << "Sim Track " << setw(2) << tp.index() << " pT: " << setw(6) << tp->pt() << " matched to " << trackV.size()
           << " reco::Tracks" << std::endl;
      for (std::vector<std::pair<RefToBase<Track>, double>>::const_iterator it = trackV.begin(); it != trackV.end();
           ++it) {
        RefToBase<Track> tr = it->first;
        double assocChi2 = it->second;
        cout << "\t\treco::Track pT: " << setw(6) << tr->pt() << " chi2: " << assocChi2 << endl;
      }
    } catch (Exception const &) {
      cout << "->   TrackingParticle " << setw(2) << tp.index() << " pT: " << setprecision(2) << setw(6) << tp->pt()
           << " matched to 0  reco::Tracks" << endl;
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_FWK_MODULE(testTrackAssociator);
