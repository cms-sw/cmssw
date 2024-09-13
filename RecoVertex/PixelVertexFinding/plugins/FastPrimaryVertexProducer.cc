// -*- C++ -*-
//
// Package:    FastPrimaryVertexProducer
// Class:      FastPrimaryVertexProducer
//
/**\class FastPrimaryVertexProducer FastPrimaryVertexProducer.cc RecoBTag/FastPrimaryVertexProducer/src/FastPrimaryVertexProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrea RIZZI
//         Created:  Thu Dec 22 14:51:44 CET 2011
//
//

// system include files
#include <memory>
#include <vector>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "CommonTools/Clustering1D/interface/Clusterizer1DCommons.h"
#include "CommonTools/Clustering1D/interface/Cluster1DMerger.h"
#include "CommonTools/Clustering1D/interface/TrivialWeightEstimator.h"
#define HaveMtv
#define HaveFsmw
#define HaveDivisive
#ifdef HaveMtv
#include "CommonTools/Clustering1D/interface/MtvClusterizer1D.h"
#endif
#ifdef HaveFsmw
#include "CommonTools/Clustering1D/interface/FsmwClusterizer1D.h"
#endif
#ifdef HaveDivisive
#include "CommonTools/Clustering1D/interface/DivisiveClusterizer1D.h"
#endif

using namespace std;

//
// class declaration
//

class FastPrimaryVertexProducer : public edm::global::EDProducer<> {
public:
  explicit FastPrimaryVertexProducer(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> const m_geomToken;
  edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> const m_pixelCPEToken;
  edm::EDGetTokenT<SiPixelClusterCollectionNew> m_clusters;
  edm::EDGetTokenT<edm::View<reco::Jet> > m_jets;
  edm::EDGetTokenT<reco::BeamSpot> m_beamSpot;
  double m_maxZ;
  double m_maxSizeX;
  double m_maxDeltaPhi;
  double m_clusterLength;
};

FastPrimaryVertexProducer::FastPrimaryVertexProducer(const edm::ParameterSet& iConfig)
    : m_geomToken(esConsumes()),
      m_pixelCPEToken(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("pixelCPE")))) {
  m_clusters = consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("clusters"));
  m_jets = consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("jets"));
  m_beamSpot = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"));
  m_maxZ = iConfig.getParameter<double>("maxZ");
  m_maxSizeX = iConfig.getParameter<double>("maxSizeX");
  m_maxDeltaPhi = iConfig.getParameter<double>("maxDeltaPhi");
  m_clusterLength = iConfig.getParameter<double>("clusterLength");
  produces<reco::VertexCollection>();
}

void FastPrimaryVertexProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  using namespace reco;
  using namespace std;

  Handle<SiPixelClusterCollectionNew> cH;
  iEvent.getByToken(m_clusters, cH);
  const SiPixelClusterCollectionNew& pixelClusters = *cH.product();

  Handle<edm::View<reco::Jet> > jH;
  iEvent.getByToken(m_jets, jH);
  const edm::View<reco::Jet>& jets = *jH.product();

  CaloJetCollection selectedJets;
  for (edm::View<reco::Jet>::const_iterator it = jets.begin(); it != jets.end(); it++) {
    if (it->pt() > 40 && fabs(it->eta()) < 1.6) {
      const CaloJet* ca = dynamic_cast<const CaloJet*>(&(*it));
      if (ca == nullptr)
        abort();
      selectedJets.push_back(*ca);
      //    std::cout << "Jet eta,phi,pt: "<< it->eta() << "," << it->phi() << "," << it->pt()   << std::endl;
    }
  }

  const PixelClusterParameterEstimator* pp = &iSetup.getData(m_pixelCPEToken);

  edm::Handle<BeamSpot> beamSpot;
  iEvent.getByToken(m_beamSpot, beamSpot);

  const TrackerGeometry* trackerGeometry = &iSetup.getData(m_geomToken);

  float lengthBmodule = 6.66;  //cm
  std::vector<float> zProjections;
  for (CaloJetCollection::const_iterator jit = selectedJets.begin(); jit != selectedJets.end(); jit++) {
    float px = jit->px();
    float py = jit->py();
    float pz = jit->pz();
    float pt = jit->pt();

    float jetZOverRho = jit->momentum().Z() / jit->momentum().Rho();
    int minSizeY = fabs(2. * jetZOverRho) - 1;
    int maxSizeY = fabs(2. * jetZOverRho) + 2;
    if (fabs(jit->eta()) > 1.6) {
      minSizeY = 1;
    }

    for (SiPixelClusterCollectionNew::const_iterator it = pixelClusters.begin(); it != pixelClusters.end();
         it++)  //Loop on pixel modules with clusters
    {
      DetId id = it->detId();
      const edmNew::DetSet<SiPixelCluster>& detset = (*it);
      Point3DBase<float, GlobalTag> modulepos = trackerGeometry->idToDet(id)->position();
      float zmodule = modulepos.z() -
                      ((modulepos.x() - beamSpot->x0()) * px + (modulepos.y() - beamSpot->y0()) * py) / pt * pz / pt;
      if ((fabs(deltaPhi(jit->momentum().Phi(), modulepos.phi())) < m_maxDeltaPhi * 2) &&
          (fabs(zmodule) < (m_maxZ + lengthBmodule / 2))) {
        for (size_t j = 0; j < detset.size(); j++)  // Loop on pixel clusters on this module
        {
          const SiPixelCluster& aCluster = detset[j];
          if (aCluster.sizeX() < m_maxSizeX && aCluster.sizeY() >= minSizeY && aCluster.sizeY() <= maxSizeY) {
            Point3DBase<float, GlobalTag> v = trackerGeometry->idToDet(id)->surface().toGlobal(
                pp->localParametersV(aCluster, (*trackerGeometry->idToDetUnit(id)))[0].first);
            GlobalPoint v_bs(v.x() - beamSpot->x0(), v.y() - beamSpot->y0(), v.z());
            if (fabs(deltaPhi(jit->momentum().Phi(), v_bs.phi())) < m_maxDeltaPhi) {
              float z = v.z() - ((v.x() - beamSpot->x0()) * px + (v.y() - beamSpot->y0()) * py) / pt * pz / pt;
              if (fabs(z) < m_maxZ) {
                zProjections.push_back(z);
              }
            }
          }  //if compatible cluster
        }    // loop on module hits
      }      // if compatible module
    }        // loop on pixel modules

  }  // loop on selected jets
  std::sort(zProjections.begin(), zProjections.end());

  std::vector<float>::iterator itCenter = zProjections.begin();
  std::vector<float>::iterator itLeftSide = zProjections.begin();
  std::vector<float>::iterator itRightSide = zProjections.begin();
  std::vector<int> counts;
  float zCluster = m_clusterLength / 2.0;  //cm
  int max = 0;
  std::vector<float>::iterator left, right;
  for (; itCenter != zProjections.end(); itCenter++) {
    while (itLeftSide != zProjections.end() && (*itCenter - *itLeftSide) > zCluster)
      itLeftSide++;
    while (itRightSide != zProjections.end() && (*itRightSide - *itCenter) < zCluster)
      itRightSide++;

    int n = itRightSide - itLeftSide;
    // std::cout << "algo :"<< *itCenter << " " << itCenter-zProjections.begin() << "  dists: " <<  (*itCenter - *itLeftSide) << " " << (*itRightSide - *itCenter) << " count: " <<  n << std::endl;
    counts.push_back(n);
    if (n > max) {
      max = n;
      left = itLeftSide;
    }
    if (n >= max) {
      max = n;
      right = itRightSide;
      //          std::cout << "algo :"<< i << " " << j << " " << *itCenter << " " << itCenter-zProjections.begin() << "  dists: " <<  (*itCenter - *itLeftSide) << " " << (*itRightSide - *itCenter) << " count: " <<  n << std::endl;
    }
  }

  float res = 0;
  if (!zProjections.empty()) {
    res = *(left + (right - left) / 2);
    //     std::cout << "RES " << res << std::endl;
    Vertex::Error e;
    e(0, 0) = 0.0015 * 0.0015;
    e(1, 1) = 0.0015 * 0.0015;
    e(2, 2) = 1.5 * 1.5;
    Vertex::Point p(beamSpot->x(res), beamSpot->y(res), res);
    Vertex thePV(p, e, 1, 1, 0);
    auto pOut = std::make_unique<reco::VertexCollection>();
    pOut->push_back(thePV);
    iEvent.put(std::move(pOut));
  } else {
    //   std::cout << "DUMMY " << res << std::endl;

    Vertex::Error e;
    e(0, 0) = 0.0015 * 0.0015;
    e(1, 1) = 0.0015 * 0.0015;
    e(2, 2) = 1.5 * 1.5;
    Vertex::Point p(beamSpot->x(res), beamSpot->y(res), res);
    Vertex thePV(p, e, 0, 0, 0);
    auto pOut = std::make_unique<reco::VertexCollection>();
    pOut->push_back(thePV);
    iEvent.put(std::move(pOut));
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(FastPrimaryVertexProducer);
