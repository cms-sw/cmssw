#ifndef ImpactParameterSelector_h
#define ImpactParameterSelector_h

//
// Original Author:  Sebastian Naumann
//         Created:  Tue Apr 20 12:45:30 CEST 2010
// $Id: ImpactParameterSelector.h,v 1.4 2010/04/21 11:52:16 snaumann Exp $
//
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

template <typename C>
class ImpactParameterSelector : public edm::EDProducer {
public:
  explicit ImpactParameterSelector(const edm::ParameterSet&);
  ~ImpactParameterSelector();

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  edm::InputTag vertices_;
  edm::InputTag leptons_;
  double cut_;

};

template<typename C>
ImpactParameterSelector<C>::ImpactParameterSelector(const edm::ParameterSet& cfg):
  vertices_(cfg.getParameter<edm::InputTag>("vertices")),
  leptons_ (cfg.getParameter<edm::InputTag>("leptons" )),
  cut_     (cfg.getParameter<double>       ("cut"     ))
{
  produces<std::vector<C> >();
}

template<typename C>
ImpactParameterSelector<C>::~ImpactParameterSelector()
{
}

template<typename C>
void
ImpactParameterSelector<C>::produce(edm::Event& event, const edm::EventSetup& setup)
{
  std::auto_ptr<std::vector<C> > out(new std::vector<C>);

  edm::Handle<reco::VertexCollection> vertices;
  event.getByLabel(vertices_, vertices);
  const reco::Vertex &vertex = *vertices->begin();

  edm::Handle<edm::View<C> > leptons; 
  event.getByLabel(leptons_, leptons);

  edm::ESHandle<TransientTrackBuilder> trackBuilder;
  setup.get<TransientTrackRecord>().get("TransientTrackBuilder", trackBuilder);

  for(typename edm::View<C>::const_iterator iter=leptons->begin(); iter!=leptons->end(); ++iter) {
    reco::TransientTrack transTrack;

    // electrons
    if(dynamic_cast<const reco::GsfElectron*>(&*iter)) {
      reco::GsfTrackRef trackRef = iter->gsfTrack();
      if(!(trackRef.isNonnull() && trackRef.isAvailable()))
	continue;
      transTrack = trackBuilder->build(trackRef);
    }
    // muons
    else {
      reco::TrackRef trackRef = iter->track();
      if(!(trackRef.isNonnull() && trackRef.isAvailable()))
	continue;
      transTrack = trackBuilder->build(trackRef);
    }

    double ipSignificance = IPTools::absoluteTransverseImpactParameter(transTrack, vertex).second.significance();
    if(ipSignificance < cut_)
      out->push_back(*iter);
  }

  event.put(out);
}

template<typename C>
void 
ImpactParameterSelector<C>::beginJob()
{
}

template<typename C>
void 
ImpactParameterSelector<C>::endJob() {
}

#endif
