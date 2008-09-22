
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopPairBSM/interface/CATopJetProducer.h"

using namespace std;
using namespace reco;
using namespace JetReco;
using namespace edm;
using namespace cms;

namespace {
  const bool debug = false;

}


CATopJetProducer::CATopJetProducer(edm::ParameterSet const& conf):
  alg_(conf.getParameter<edm::InputTag>("src"),                // calo tower collection source
       conf.getParameter<int>("algorithm"),                    // 0 = KT, 1 = CA, 2 = anti-KT
       conf.getParameter<double>("seedThreshold"),             // calo tower seed threshold
       conf.getParameter<double>("centralEtaCut"),             // eta for defining "central" jets
       conf.getParameter<double>("sumEtEtaCut"),               // eta for event SumEt
       conf.getParameter<double>("ptMin"),                     // lower pt cut on which jets to reco
       conf.getParameter<double>("etFrac"),                    // fraction of event sumEt / 2 for a jet to be considered "hard"
       conf.getParameter<bool>  ("useAdjacency"),              // veto adjacent subjets
       conf.getParameter<bool>  ("useMaxTower"),               // use max tower as adjacency criterion, otherwise use centroid
       conf.getParameter<std::vector<double> >("ptBins"),      // pt bins over which cuts vary
       conf.getParameter<std::vector<double> >("rBins"),       // cone size bins,
       conf.getParameter<std::vector<double> >("ptFracBins"),  // fraction of hard jet that subjet must have
       conf.getParameter<std::vector<int> >("nCellBins")       // number of cells to consider two subjets adjacent
       )
{
  produces<reco::CaloJetCollection>("caTopSubJets").setBranchAlias("caTopSubJets");
  produces<reco::BasicJetCollection>();
}
  
void CATopJetProducer::produce( edm::Event & e, const edm::EventSetup & c ) {
  
  alg_.run( e, c );
  
}

  
//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetProducer);
