#ifndef __TMTrackTrigger_VertexFinder_VertexProducer_h__
#define __TMTrackTrigger_VertexFinder_VertexProducer_h__


#include <map>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


namespace l1tVertexFinder {
  // class Histos;
  class Settings;
}

class VertexProducer : public edm::EDProducer {

public:
  explicit VertexProducer(const edm::ParameterSet&);
  ~VertexProducer(){}

private:

  typedef edm::View< TTTrack< Ref_Phase2TrackerDigi_ > > TTTrackCollectionView;

  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

private:
  const edm::EDGetTokenT<TTTrackCollectionView> l1TracksToken_;

  l1tVertexFinder::Settings *settings_;
};

#endif

