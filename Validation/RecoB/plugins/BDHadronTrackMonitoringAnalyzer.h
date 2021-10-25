#ifndef BDHadronTrackMonitoringAnalyzer_H
#define BDHadronTrackMonitoringAnalyzer_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "SimTracker/TrackHistory/interface/TrackCategories.h"
#include "SimTracker/TrackHistory/interface/TrackClassifier.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMOffline/RecoB/interface/Tools.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <fstream>
#include <iostream>

/** \class BDHadronTrackMonitoringAnalyzer
 *
 *  Top level steering routine for B + D hadron track monitoring tool from
 * RECODEBUG samples.
 *
 */

class BDHadronTrackMonitoringAnalyzer : public DQMEDAnalyzer {
public:
  explicit BDHadronTrackMonitoringAnalyzer(const edm::ParameterSet &pSet);

  ~BDHadronTrackMonitoringAnalyzer() override;

  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  enum HistoryClasses { BCWeakDecay = 0, BWeakDecay = 1, CWeakDecay = 2, PU = 3, Other = 4, Fake = 5 };
  static const std::vector<std::string> TrkHistCat;

private:
  // cut values
  double distJetAxis_;
  double decayLength_;
  double minJetPt_;
  double maxJetEta_;

  // strings
  std::string ipTagInfos_;

  // InputTags
  edm::InputTag PatJetSrc_;
  edm::InputTag TrackSrc_;
  edm::InputTag PVSrc_;
  edm::InputTag ClusterTPMapSrc_;

  // Tokens
  edm::EDGetTokenT<pat::JetCollection> PatJetCollectionTag_;
  edm::EDGetTokenT<reco::TrackCollection> TrackCollectionTag_;
  edm::EDGetTokenT<reco::VertexCollection> PrimaryVertexColl_;
  edm::EDGetTokenT<ClusterTPAssociation> clusterTPMapToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> ttrackToken_;
  // TrackClassifier
  TrackClassifier classifier_;

  // Histograms
  // b jets
  MonitorElement *nTrkAll_bjet;  // total number of selected tracks (or TrackingParticles)
  MonitorElement *nTrk_bjet[6];  // total number of selected tracks (or TrackingParticles)
                                 // in each TrackHistory category
  // c jets
  MonitorElement *nTrkAll_cjet;  // total number of selected tracks (or TrackingParticles)
  MonitorElement *nTrk_cjet[6];  // total number of selected tracks (or TrackingParticles)
                                 // in each TrackHistory category
  // dusg jets
  MonitorElement *nTrkAll_dusgjet;  // total number of selected tracks (or TrackingParticles)
  MonitorElement *nTrk_dusgjet[6];  // total number of selected tracks (or
                                    // TrackingParticles) in each TrackHistory category

  // track properties for all flavours combined
  MonitorElement *TrkPt_alljets[6],
      *TrkTruthPt_alljets[5];  // Pt of selected tracks (or TrackingParticles)
  MonitorElement *TrkEta_alljets[6],
      *TrkTruthEta_alljets[5];  // Eta of selected tracks (or TrackingParticles)
  MonitorElement *TrkPhi_alljets[6],
      *TrkTruthPhi_alljets[5];  // Phi of selected tracks (or TrackingParticles)
  MonitorElement *TrkDxy_alljets[6],
      *TrkTruthDxy_alljets[5];  // Transverse IP of selected tracks (or
                                // TrackingParticles)
  MonitorElement *TrkDz_alljets[6],
      *TrkTruthDz_alljets[5];  // Longitudinal IP of selected tracks (or
                               // TrackingParticles)
  MonitorElement *TrkHitAll_alljets[6],
      *TrkTruthHitAll_alljets[5];  // total number Tracker hits of selected
                                   // tracks (or TrackingParticles)
  MonitorElement *TrkHitStrip_alljets[6],
      *TrkTruthHitStrip_alljets[5];  // number of strip hits of of selected
                                     // tracks (or TrackingParticles)
  MonitorElement *TrkHitPixel_alljets[6],
      *TrkTruthHitPixel_alljets[5];  // number of pixel hits of selected tracks
                                     // (or TrackingParticles)
};

#endif
