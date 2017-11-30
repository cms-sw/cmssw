#ifndef __TMTrackTrigger_VertexFinder_VertexProducer_h__
#define __TMTrackTrigger_VertexFinder_VertexProducer_h__


#include <map>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"



namespace vertexFinder {
  class Histos;
  class Settings;
}

class VertexProducer : public edm::EDProducer {

public:
  explicit VertexProducer(const edm::ParameterSet&);	
  ~VertexProducer(){}

private:

  typedef std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > TTTrackCollection;

  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

private:
  const edm::EDGetTokenT<TrackingParticleCollection> tpInputTag;
  typedef edmNew::DetSetVector< TTStub<Ref_Phase2TrackerDigi_> > DetSetVec;
  const edm::EDGetTokenT<DetSetVec> stubInputTag;
  typedef TTStubAssociationMap<Ref_Phase2TrackerDigi_>           TTStubAssMap;
  const edm::EDGetTokenT<TTStubAssMap> stubTruthInputTag;
  typedef TTClusterAssociationMap<Ref_Phase2TrackerDigi_>        TTClusterAssMap;
  const edm::EDGetTokenT<TTClusterAssMap> clusterTruthInputTag;
  const edm::EDGetTokenT<TTTrackCollection> l1TracksToken_;

  vertexFinder::Settings *settings_;
  vertexFinder::Histos   *hists_;
};

#endif

