
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
  src_    (conf.getParameter<edm::InputTag>("src")),               // input collection
  alg_(src_,
       conf.getParameter<int>("algorithm"),                    // 0 = KT, 1 = CA, 2 = anti-KT
       conf.getParameter<double>("seedThreshold"),             // calo tower seed threshold
       conf.getParameter<double>("centralEtaCut"),             // eta for defining "central" jets
       conf.getParameter<double>("sumEtEtaCut"),               // eta for event SumEt
       conf.getParameter<double>("ptMin"),                     // lower pt cut on which jets to reco
       conf.getParameter<double>("etFrac"),                    // fraction of event sumEt / 2 for a jet to be considered "hard"
       conf.getParameter<bool>  ("useAdjacency"),              // veto adjacent subjets
       conf.getParameter<bool>  ("useMaxTower"),               // use max tower as adjacency criterion, otherwise use centroid
       conf.getParameter<std::vector<double> > ("ptBins"),     // pt bins over which cuts may vary
       conf.getParameter<std::vector<double> >("rBins"),       // cone size bins,
       conf.getParameter<std::vector<double> >("ptFracBins"),  // fraction of hard jet that subjet must have
       conf.getParameter<std::vector<int> >("nCellBins")       // number of cells to consider two subjets adjacent
       )
{
  produces<reco::CaloJetCollection>("caTopSubJets").setBranchAlias("caTopSubJets");
  produces<reco::BasicJetCollection>();
}
  
void CATopJetProducer::produce( edm::Event & e, const edm::EventSetup & c ) 
{

  bool verbose = false;

  // get a list of output subjets
  std::auto_ptr<CaloJetCollection>  subjetCollection( new CaloJetCollection() );
  // get a list of output jets
  std::auto_ptr<BasicJetCollection>  jetCollection( new BasicJetCollection() );
  // this is the mapping of subjet to hard jet
  std::vector< std::vector<int> > indices;
  // this is the list of hardjet 4-momenta
  std::vector<math::XYZTLorentzVector> p4_hardJets;

  // get calo towers from event record

  Handle<CaloTowerCollection> fInputHandle;
  e.getByLabel( src_, fInputHandle );

  CaloTowerCollection const & fInput = *fInputHandle;

  // list of fastjet pseudojet constituents
  vector<fastjet::PseudoJet> cell_particles;
  CaloTowerCollection::const_iterator input = fInput.begin();
  if ( verbose ) cout << "Adding cell particles, n = " << fInput.size()  << endl;
  for (unsigned i = 0; i < fInput.size(); ++i) {
    const CaloTower & c = fInput[i];
    cell_particles.push_back (fastjet::PseudoJet (c.px(),c.py(),c.pz(),c.energy()));
    cell_particles.back().set_user_index(i);
  }

  // Sort by et
  GreaterByEtPseudoJet compEt;
  sort( cell_particles.begin(), cell_particles.end(), compEt );


  vector<CATopPseudoJet> outputs;
  
  alg_.run( cell_particles, outputs, c );


  vector<CATopPseudoJet>::const_iterator it = outputs.begin(),
    iEnd = outputs.end(),
    iBegin = outputs.begin();

  indices.resize( outputs.size() );

  for ( ; it != iEnd; ++it ) {

    int jetIndex = it - iBegin;
    fastjet::PseudoJet localJet = it->hardJet();

    p4_hardJets.push_back( math::XYZTLorentzVector(localJet.px(), localJet.py(), localJet.pz(), localJet.e() ));


    // create the subjets
    std::vector<CATopPseudoSubJet>::const_iterator itSubJetBegin = it->subjets().begin(),
      itSubJet = itSubJetBegin, itSubJetEnd = it->subjets().end();
    for (; itSubJet != itSubJetEnd; ++itSubJet ){

      fastjet::PseudoJet subjet = itSubJet->subjet();
      math::XYZTLorentzVector p4Subjet(subjet.px(), subjet.py(), subjet.pz(), subjet.e() );
      reco::Particle::Point point(0,0,0);

      // Find the subjet constituents
      vector<CandidatePtr> subjetConstituents;

      // Get the transient subjet constituents from fastjet
      vector<int> const & subjetFastjetConstituentIndices = itSubJet->constituents();

      vector<int>::const_iterator fastSubIt = subjetFastjetConstituentIndices.begin(),
	transConstBegin = subjetFastjetConstituentIndices.begin(),
	transConstEnd = subjetFastjetConstituentIndices.end();
      for ( ; fastSubIt != transConstEnd; ++fastSubIt ) {

	if ( *fastSubIt >= 0 && *fastSubIt < cell_particles.size() ) 
	  subjetConstituents.push_back(CandidatePtr(fInputHandle, *fastSubIt));
      }

      // This holds the subjet-to-hardjet mapping
      indices[jetIndex].push_back( subjetCollection->size() );      
      // Add the constituent to the subjet to write to event record
      subjetCollection->push_back( CaloJet(p4Subjet, point, CaloJet::Specific(), subjetConstituents) );

    }



  }


  if ( verbose ) cout << "List of subjets" << endl;
  CaloJetCollection::const_iterator subjetPrintIt = subjetCollection->begin(),
    subjetPrintEnd = subjetCollection->end() ;
  for ( ; subjetPrintIt != subjetPrintEnd; ++subjetPrintIt ) {
    char buff[800];
    sprintf(buff, "CaloJet pt = %6.2f, eta = %6.2f, phi = %6.2f", 
	    subjetPrintIt->pt(), subjetPrintIt->eta(), subjetPrintIt->phi() );
    if ( verbose ) cout << buff << endl;
  }

  // put subjets into event record
  if ( verbose ) cout << "About to put subjets, size = " << subjetCollection->size() << endl;
  OrphanHandle<CaloJetCollection> subjetHandleAfterPut = e.put( subjetCollection, "caTopSubJets" );

  
  // Now create the hard jets
  if ( verbose ) cout << "About to make hard jets for writing" << endl;
  vector<math::XYZTLorentzVector>::const_iterator ip4 = p4_hardJets.begin(),
    ip4Begin = p4_hardJets.begin(),
    ip4End = p4_hardJets.end();

  for ( ; ip4 != ip4End; ++ip4 ) {
    int p4_index = ip4 - ip4Begin;
    vector<int> & ind = indices[p4_index];
    vector<CandidatePtr> i_hardJetConstituents;
    // Add the subjets to the hard jet
    for( vector<int>::const_iterator isub = ind.begin();
	 isub != ind.end(); ++isub ) {
      CandidatePtr candPtr( subjetHandleAfterPut, *isub, false );
      i_hardJetConstituents.push_back( candPtr );
    }   
    reco::Particle::Point point(0,0,0);
    jetCollection->push_back( BasicJet( *ip4, point, i_hardJetConstituents) );
  }
  

  // put hard jets into event record
  if ( verbose ) cout << "About to put hard jets, size = " << jetCollection->size() << endl;
  e.put( jetCollection);
  
  if ( verbose ) cout << "Done" << endl;



  
}

  
//define this as a plug-in
DEFINE_FWK_MODULE(CATopJetProducer);
