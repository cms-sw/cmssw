/** \class MuonDetCleaner
 *
 * Clean collections of hits in muon detectors (CSC, DT and RPC)
 * for original Zmumu event and "embedded" simulated tau decay products
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 *
 *
 * Clean Up from STefan Wayand, KIT
 *
 */
#ifndef TauAnalysis_MCEmbeddingTools_MuonDetCleaner_H
#define TauAnalysis_MCEmbeddingTools_MuonDetCleaner_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Transition.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <map>
#include <string>
#include <vector>

template <typename T1, typename T2>
class MuonDetCleaner : public edm::stream::EDProducer<> {
public:
  explicit MuonDetCleaner(const edm::ParameterSet &);
  ~MuonDetCleaner() override;

private:
  typedef edm::RangeMap<T1, edm::OwnVector<T2>> RecHitCollection;

  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void fillVetoHits(const TrackingRecHit &, std::vector<uint32_t> *);
  uint32_t getRawDetId(const T2 &);
  bool checkrecHit(const TrackingRecHit &);

  const edm::EDGetTokenT<edm::View<pat::Muon>> mu_input_;

  std::map<std::string, edm::EDGetTokenT<RecHitCollection>> inputs_;

  TrackAssociatorParameters parameters_;
  TrackDetectorAssociator trackAssociator_;
  edm::EDGetTokenT<DTDigiCollection> m_dtDigisToken;
  edm::EDGetTokenT<CSCStripDigiCollection> m_cscDigisToken;
  edm::Handle<DTDigiCollection> m_dtDigis;
  edm::Handle<CSCStripDigiCollection> m_cscDigis;
  edm::ESHandle<DTGeometry> m_dtGeometry;
  edm::ESHandle<CSCGeometry> m_cscGeometry;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> m_dtGeometryToken;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> m_cscGeometryToken;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  double m_digiMaxDistanceX;
};

#endif
