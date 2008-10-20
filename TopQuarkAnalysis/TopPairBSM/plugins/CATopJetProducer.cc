
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TopQuarkAnalysis/TopPairBSM/interface/CATopJetProducer.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace reco;
using namespace JetReco;
using namespace edm;
using namespace cms;

namespace {
  const bool debug = false;

}


CATopJetProducer::CATopJetProducer(edm::ParameterSet const& conf):
  src_       (conf.getParameter<edm::InputTag>("src")),              // input collection
  jetType_   (conf.getUntrackedParameter<std::string>  ("jetType")), // jet reconstruction type
  alg_(src_,
       conf.getParameter<int>("algorithm"),                    // 0 = KT, 1 = CA, 2 = anti-KT
       conf.getParameter<double>("inputEtMin"),                // seed threshold
       conf.getParameter<double>("centralEtaCut"),             // eta for defining "central" jets
       conf.getParameter<double>("sumEtEtaCut"),               // eta for event SumEt
       conf.getParameter<double>("jetPtMin"),                  // lower pt cut on which jets to reco
       conf.getParameter<double>("etFrac"),                    // fraction of event sumEt / 2 for a jet to be considered "hard"
       conf.getParameter<bool>  ("useAdjacency"),              // veto adjacent subjets
       conf.getParameter<bool>  ("useMaxTower"),               // use max tower as adjacency criterion, otherwise use centroid
       conf.getParameter<std::vector<double> > ("ptBins"),     // pt bins over which cuts may vary
       conf.getParameter<std::vector<double> >("rBins"),       // cone size bins,
       conf.getParameter<std::vector<double> >("ptFracBins"),  // fraction of hard jet that subjet must have
       conf.getParameter<std::vector<int> >("nCellBins")       // number of cells to consider two subjets adjacent
       )
{
  if ( jetType_ == "CaloJet" )
    produces<reco::CaloJetCollection>("caTopSubJets").setBranchAlias("caTopSubJets");
  else if ( jetType_ == "GenJet" )
    produces<reco::GenJetCollection>("caTopSubJets").setBranchAlias("caTopSubJets");
  else if ( jetType_ == "BasicJet" )
    produces<reco::BasicJetCollection>("caTopSubJets").setBranchAlias("caTopSubJets");
  else if ( jetType_ == "PFJet" )
    produces<reco::PFJetCollection>("caTopSubJets").setBranchAlias("caTopSubJets");
  else {
    throw cms::Exception("Invalid input type") << "Input type for CATopJetProducer is invalid\n";
  }


  produces<reco::BasicJetCollection>();
}
  
void CATopJetProducer::produce( edm::Event & e, const edm::EventSetup & c ) 
{

  bool verbose = false;

 
  // -------------------------------------------------------
  // Set up the constituent list
  // -------------------------------------------------------

  // list of fastjet pseudojet constituents
  vector<fastjet::PseudoJet> cell_particles;


  // get input from event record
  Handle<View<Candidate> > fInputHandle;
  e.getByLabel( src_, fInputHandle );

  View<Candidate> const & fInput = *fInputHandle;

  // Fill list of fastjet pseudojet constituents
  View<Candidate>::const_iterator inputIt = fInput.begin(),
    inputEnd = fInput.end(),
    inputBegin = inputIt;
  if ( verbose ) cout << "Adding cell particles, n = " << fInput.size()  << endl;

  for ( ; inputIt != inputEnd; ++inputIt ) {
    cell_particles.push_back (fastjet::PseudoJet (inputIt->px(),inputIt->py(),inputIt->pz(),inputIt->energy()));
    cell_particles.back().set_user_index(inputIt - inputBegin);
  }

  // Sort pseudojets by et
  GreaterByEtPseudoJet compEt;
  sort( cell_particles.begin(), cell_particles.end(), compEt );

  // Here is the list of pseudojet "hard + soft" jets in a structure
  vector<CATopPseudoJet> outputs;
  



  // -------------------------------------------------------
  // Run the actual algorithm  
  // -------------------------------------------------------
  alg_.run( cell_particles, outputs, c );

  // -------------------------------------------------------
  // Now fill the outputs
  // -------------------------------------------------------

  if ( jetType_ == "CaloJet" )
    write_outputs<reco::CaloJet>( e, c, outputs, fInputHandle ); 
  else if ( jetType_ == "GenJet" )
    write_outputs<reco::GenJet>( e, c, outputs, fInputHandle ); 
  else if ( jetType_ == "BasicJet" )
    write_outputs<reco::BasicJet>( e, c, outputs, fInputHandle ); 
  else if ( jetType_ == "PFJet" )
    write_outputs<reco::PFJet>( e, c, outputs, fInputHandle ); 




}



  
//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetProducer);
