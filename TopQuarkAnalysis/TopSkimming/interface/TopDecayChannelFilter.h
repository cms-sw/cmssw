#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

template <typename S>
class TopDecayChannelFilter : public edm::EDFilter {
 public:
  TopDecayChannelFilter(const edm::ParameterSet&);
  ~TopDecayChannelFilter();

 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  edm::InputTag src_;
  edm::EDGetTokenT<TtGenEvent> genEvt_;
  edm::EDGetTokenT<reco::GenParticleCollection> parts_;
  S sel_;
  bool checkedSrcType_;
  bool useTtGenEvent_;
};

template<typename S>
TopDecayChannelFilter<S>::TopDecayChannelFilter(const edm::ParameterSet& cfg):
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ),
  genEvt_( mayConsume<TtGenEvent>( src_ ) ),
  parts_( mayConsume<reco::GenParticleCollection>( src_ ) ),
  sel_( cfg ),
  checkedSrcType_(0), useTtGenEvent_(0)
{ }

template<typename S>
TopDecayChannelFilter<S>::~TopDecayChannelFilter()
{ }

template<typename S>
bool
TopDecayChannelFilter<S>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::GenParticleCollection> parts;
  edm::Handle<TtGenEvent> genEvt;

  if(!checkedSrcType_) {
    checkedSrcType_ = true;
    if(iEvent.getByToken( genEvt_, genEvt )) {
      useTtGenEvent_ = true;
      return sel_( genEvt->particles(), src_.label() );
    }
  }
  else {
    if(useTtGenEvent_) {
      iEvent.getByToken( genEvt_, genEvt );
      return sel_( genEvt->particles(), src_.label() );
    }
  }
  iEvent.getByToken( parts_, parts );
  return sel_( *parts, src_.label() );
}
