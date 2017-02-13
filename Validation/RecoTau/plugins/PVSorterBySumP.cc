////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <memory>
#include <vector>
#include <sstream>


////////////////////////////////////////////////////////////////////////////////
// class definition
////////////////////////////////////////////////////////////////////////////////
template<typename T1>
class bestPVselector : public edm::EDProducer {
public:

  explicit bestPVselector(edm::ParameterSet const& iConfig);
  void produce(edm::Event&, edm::EventSetup const&) override;

private:

  edm::EDGetTokenT<std::vector<T1>> src_;
};


////////////////////////////////////////////////////////////////////////////////
// construction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T1>
bestPVselector<T1>::bestPVselector(edm::ParameterSet const& iConfig)
  : src_{consumes<std::vector<T1>>(iConfig.getParameter<edm::InputTag>("src"))}
{
  produces<std::vector<T1>>();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
template<typename T1>
void bestPVselector<T1>::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  edm::Handle<std::vector<T1>> vertices;
  iEvent.getByToken(src_,vertices);

  auto theBestPV = std::make_unique<std::vector<T1>>();

  if(!vertices->empty()) {
    auto sumSquarePt = [](auto const& pv) { return pv.p4().pt()*pv.p4().pt(); };
    auto bestPV = std::max_element(std::cbegin(*vertices), std::cend(*vertices),
                                   [sumSquarePt](auto const& v1, auto const& v2) {
                                     return sumSquarePt(v1) < sumSquarePt(v2);
                                   });
    theBestPV->push_back(*bestPV);
  }
  iEvent.put(std::move(theBestPV));
}

using HighestSumP4PrimaryVertexSelector = bestPVselector<reco::Vertex>;
DEFINE_FWK_MODULE(HighestSumP4PrimaryVertexSelector);
