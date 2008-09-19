// Original author: Brock Tweedie (JHU)
// Ported to CMSSW by: Sal Rappoccio (JHU)
// $Id: CATopJetAlgorithm.cc,v 1.1 2008/09/05 15:03:32 srappocc Exp $

#include "TopQuarkAnalysis/TopPairBSM/interface/CATopJetAlgorithm.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"


using namespace std;
using namespace reco;
using namespace JetReco;
using namespace edm;



ostream & operator<<(ostream & out, CaloJet & j)
{
  char buff[800];
  sprintf(buff, "CaloJet pt = %6.2f, eta = %6.2f, phi = %6.2f", 
	  j.pt(), j.eta(), j.phi() );
  out << buff;
  return out;
}

ostream & operator<<(ostream & out, fastjet::PseudoJet j)
{
  char buff[800];
  sprintf(buff, "PseudoJet index = %3d, pt = %6.2f, eta = %6.2f, phi = %6.2f", 
	  j.user_index(), j.perp(), j.eta(), j.phi() );
  out << buff;
  return out;
}

class GreaterByEtPseudoJet : 
  public std::binary_function<fastjet::PseudoJet const &, fastjet::PseudoJet const &, bool> {

public:
  bool operator()( fastjet::PseudoJet const & j1, fastjet::PseudoJet const & j2 ) {
    return j1.perp() > j2.perp();
  }
};

//  Run the algorithm
//  ------------------
void CATopJetAlgorithm::run( edm::Event & e, const edm::EventSetup & c )  
{

  // Get the calo geometry
  edm::ESHandle<CaloGeometry> geometry;
  c.get<CaloGeometryRecord>().get(geometry);
  const CaloSubdetectorGeometry* towerGeometry = 
    geometry->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId);  

  


  // get a list of output subjets
  std::auto_ptr<CaloJetCollection>  subjetCollection( new CaloJetCollection() );
  // get a list of output jets
  std::auto_ptr<BasicJetCollection>  jetCollection( new BasicJetCollection() );
    

  bool verbose = false;

  if ( verbose ) cout << "Welcome to CATopSubJetAlgorithm::run" << endl;

  // get calo towers from event record

  Handle<CaloTowerCollection> fInputHandle;
  e.getByLabel( mSrc_, fInputHandle );

  CaloTowerCollection const & fInput = *fInputHandle;


  //make a list of input objects ordered by ET and calculate sum et

  // Sum Et of the event
  double sumEt = 0.;


  // list of fastjet pseudojet constituents
  vector<fastjet::PseudoJet> cell_particles;
  CaloTowerCollection::const_iterator input = fInput.begin();
  if ( verbose ) cout << "Adding cell particles, n = " << fInput.size()  << endl;
  for (unsigned i = 0; i < fInput.size(); ++i) {
    sumEt += fInput[i].et();
    const CaloTower & c = fInput[i];
    cell_particles.push_back (fastjet::PseudoJet (c.px(),c.py(),c.pz(),c.energy()));
    cell_particles.back().set_user_index(i);
    if ( verbose ) cout << "Adding cell particle " << cell_particles.back() << endl;
  }


  if ( verbose ) cout << "Sorting by pt" << endl;
  // Sort by et
  GreaterByEtPseudoJet compEt;
  sort( cell_particles.begin(), cell_particles.end(), compEt );


  // empty 4-vector
  fastjet::PseudoJet blankJetA(0,0,0,0);
  blankJetA.set_user_index(-1);
  const fastjet::PseudoJet blankJet = blankJetA;

  // Determine which bin we are in for et clustering
  int iPt = -1;
  for ( unsigned int i = 0; i < ptBins_.size(); ++i ) {
    if ( sumEt / 2.0 > ptBins_[i] ) iPt = i;
  }
  if ( verbose ) cout << "Using sumEt = " << sumEt << ", bin = " << iPt << endl;


  // If the sum et is too low, exit
  if ( iPt < 0 ) {


    if ( verbose ) cout << "Pt is too small, exiting" << endl;

    e.put( subjetCollection, "caTopSubJets"); 
    e.put( jetCollection);
    
    return;
  }

  int nCellMin = nCellBins_[iPt];

  if ( verbose ) cout << "Using nCellMin = " << nCellMin << endl;


  // Define strategy, recombination scheme, and jet definition
  fastjet::Strategy strategy = fastjet::Best;
  fastjet::RecombinationScheme recombScheme = fastjet::E_scheme;

  // pick algorithm
  fastjet::JetAlgorithm algorithm = static_cast<fastjet::JetAlgorithm>( algorithm_ );

  fastjet::JetDefinition jetDef( algorithm, 
				 rBins_[iPt], recombScheme, strategy);   // Cambridge/Aachen

  if ( verbose ) cout << "About to do jet clustering in CA" << endl;
  // run the jet clustering.  pick out the two leading, central jets
  fastjet::ClusterSequence clusterSeq(cell_particles, jetDef);

  if ( verbose ) cout << "Getting inclusive jets" << endl;
  // Get the transient inclusive jets
  vector<fastjet::PseudoJet> inclusiveJets = clusterSeq.inclusive_jets(ptMin_);

  if ( verbose ) cout << "Getting central jets" << endl;
  // Find the transient central jets
  vector<fastjet::PseudoJet> centralJets;
  for (unsigned int i = 0; i < inclusiveJets.size(); i++) {
    if (inclusiveJets[i].perp() > ptMin_ && fabs(inclusiveJets[i].eta()) < centralEtaCut_) {
      centralJets.push_back(inclusiveJets[i]);
      if ( verbose ) cout << "Added Central Jet " << i << " = " << centralJets.back() << endl;
    }
  }
  // Sort the transient central jets in Et
  sort( centralJets.begin(), centralJets.end(), compEt );


  // These will store the 4-vectors of each hard jet
  vector<math::XYZTLorentzVector> p4_hardJets;

  // These will store the indices of each subjet that 
  // are present in each jet
  vector<vector<int> > indices( centralJets.size() );
  // This is the offset in the list of subjets for each jet
  int offset = 0; 

  // Loop over central jets, attempt to find substructure
  vector<fastjet::PseudoJet>::iterator jetIt = centralJets.begin(),
    centralJetsBegin = centralJets.begin(),
    centralJetsEnd = centralJets.end();
  for ( ; jetIt != centralJetsEnd; ++jetIt ) {

    // Get the jet index
    int jetIndex = jetIt - centralJetsBegin;

    if ( verbose ) cout << "---------------------" << endl << "Adding central jet " << *jetIt << endl;
    fastjet::PseudoJet localJet = *jetIt;

    // Get the 4-vector for this jet
    p4_hardJets.push_back( math::XYZTLorentzVector(localJet.px(), localJet.py(), localJet.pz(), localJet.e() ));

    // jet decomposition.  try to find 3 or 4 hard, well-localized subjets, characteristic of a boosted top.
    double ptHard = ptFracBins_[iPt]*localJet.perp();
    vector<fastjet::PseudoJet> leftoversAll;

    // stage 1:  primary decomposition.  look for when the jet declusters into two hard subjets
    if ( verbose ) cout << "Doing decomposition 1" << endl;
    fastjet::PseudoJet ja, jb;
    vector<fastjet::PseudoJet> leftovers1;
    bool hardBreak1 = decomposeJet(localJet,clusterSeq,fInput,towerGeometry,ptHard,nCellMin,ja,jb,leftovers1);
    leftoversAll.insert(leftoversAll.end(),leftovers1.begin(),leftovers1.end());
	
    // stage 2:  secondary decomposition.  look for when the hard subjets found above further decluster into two hard sub-subjets
    //
    // ja -> jaa+jab ?
    if ( verbose ) cout << "Doing decomposition 2" << endl;
    fastjet::PseudoJet jaa, jab;
    vector<fastjet::PseudoJet> leftovers2a;
    bool hardBreak2a = false;
    if (hardBreak1)  hardBreak2a = decomposeJet(ja,clusterSeq,fInput,towerGeometry,ptHard,nCellMin,jaa,jab,leftovers2a);
    leftoversAll.insert(leftoversAll.end(),leftovers2a.begin(),leftovers2a.end());
    // jb -> jba+jbb ?
    fastjet::PseudoJet jba, jbb;
    vector<fastjet::PseudoJet> leftovers2b;
    bool hardBreak2b = false;
    if (hardBreak1)  hardBreak2b = decomposeJet(jb,clusterSeq,fInput,towerGeometry,ptHard,nCellMin,jba,jbb,leftovers2b);
    leftoversAll.insert(leftoversAll.end(),leftovers2b.begin(),leftovers2b.end());

    // NOTE:  it might be good to consider some checks for whether these subjets can be further decomposed.  e.g., the above procedure leaves
    //        open the possibility of "subjets" that actually consist of two or more distinct hard clusters.  however, this kind of thing
    //        is a rarity for the simulations so far considered.

    // proceed if one or both of the above hard subjets successfully decomposed
    if ( verbose ) cout << "Done with decomposition" << endl;

    int nBreak2 = 0;
    fastjet::PseudoJet hardA = blankJet, hardB = blankJet, hardC = blankJet, hardD = blankJet;
    if ( hardBreak2a && !hardBreak2b) { nBreak2 = 1; hardA = jaa; hardB = jab; hardC = jb;  hardD = blankJet; }
    if (!hardBreak2a &&  hardBreak2b) { nBreak2 = 1; hardA = jba; hardB = jbb; hardC = ja;  hardD = blankJet;}
    if ( hardBreak2a &&  hardBreak2b) { nBreak2 = 2; hardA = jaa; hardB = jab; hardC = jba; hardD = jbb; }

    // check if we are left with >= 3 hard subjets
    fastjet::PseudoJet subjet1 = blankJet;
    fastjet::PseudoJet subjet2 = blankJet;
    fastjet::PseudoJet subjet3 = blankJet;
    fastjet::PseudoJet subjet4 = blankJet;
    if (nBreak2 >= 1) {
      subjet1 = hardA; subjet2 = hardB; subjet3 = hardC; subjet4 = hardD;
    }

    // record the hard subjets
    vector<fastjet::PseudoJet> hardSubjets;
    
    if ( subjet1.user_index() >= 0 )
      hardSubjets.push_back(subjet1);
    if ( subjet2.user_index() >= 0 )
      hardSubjets.push_back(subjet2);
    if ( subjet3.user_index() >= 0 )
      hardSubjets.push_back(subjet3);
    if ( subjet4.user_index() >= 0 )
      hardSubjets.push_back(subjet4);
    sort(hardSubjets.begin(), hardSubjets.end(), compEt );

    // create the subjets
    std::vector<fastjet::PseudoJet>::const_iterator itSubJetBegin = hardSubjets.begin(),
      itSubJet = itSubJetBegin, itSubJetEnd = hardSubjets.end();
    for (; itSubJet != itSubJetEnd; ++itSubJet ){
      int index = itSubJet - itSubJetBegin;
      if ( verbose ) cout << "Adding subjet " << *itSubJet << endl;
      //       if ( verbose ) cout << "Adding input collection element " << (*itSubJet).user_index() << endl;
      //       if ( (*itSubJet).user_index() >= 0 && (*itSubJet).user_index() < fInput.size() )
      
      math::XYZTLorentzVector p4Subjet(itSubJet->px(), itSubJet->py(), itSubJet->pz(), itSubJet->e() );
      reco::Particle::Point point(0,0,0);

      // Find the subjet constituents
      vector<CandidatePtr> subjetConstituents;

      // Get the transient subjet constituents from fastjet
      vector<fastjet::PseudoJet> subjetFastjetConstituents = clusterSeq.constituents( *itSubJet );

      vector<fastjet::PseudoJet>::const_iterator fastSubIt = subjetFastjetConstituents.begin(),
	transConstBegin = subjetFastjetConstituents.begin(),
	transConstEnd = subjetFastjetConstituents.end();
      for ( ; fastSubIt != transConstEnd; ++fastSubIt ) {

	if ( fastSubIt->user_index() >= 0 && fastSubIt->user_index() < fInput.size() ) 
	  subjetConstituents.push_back(CandidatePtr(fInputHandle, fastSubIt->user_index()));
      }

      indices[jetIndex].push_back( subjetCollection->size() );      
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




//-----------------------------------------------------------------------
// determine whether two clusters (made of calorimeter towers) are living on "adjacent" cells.  if they are, then
// we probably shouldn't consider them to be independent objects!
//
// From Sal: Ignoring genjet case
//
bool CATopJetAlgorithm::adjacentCells(const fastjet::PseudoJet & jet1, const fastjet::PseudoJet & jet2, 
				      const CaloTowerCollection & fInput,
				      const CaloSubdetectorGeometry  * fTowerGeometry,
				      const fastjet::ClusterSequence & theClusterSequence,
				      int nCellMin ) const {

  // Get each tower (depending on user input, can be either max pt tower, or centroid).
  CaloTowerDetId tower1 = getCaloTower( jet1, fInput, fTowerGeometry, theClusterSequence );
  CaloTowerDetId tower2 = getCaloTower( jet2, fInput, fTowerGeometry, theClusterSequence );

  // Get the number of calo towers away that the two towers are.
  // Can be non-integral fraction if "centroid" case is chosen.
  int distance = getDistance( tower1, tower2, fTowerGeometry );

  if ( distance <= nCellMin ) return true;
  else return false;
}

//-------------------------------------------------------------------------
// Find the highest pt tower inside the jet
fastjet::PseudoJet CATopJetAlgorithm::getMaxTower( const fastjet::PseudoJet & jet,
						   const CaloTowerCollection & fInput,
						   const fastjet::ClusterSequence & theClusterSequence
						   ) const
{
  // If this jet is a calo tower itself, return it
  if ( jet.user_index() > 0 ) return jet;
  // Check for the bug in fastjet where it sets the user_index to 0 instead of -1
  // in the clustering, since it might actually BE zero. 
  else if (  jet.user_index() == 0 && jet.perp() > 0 && jet.perp() == fInput[0].pt() ) {
    return jet;
  }
  // Otherwise, search through the constituents and find the highest pt tower, return it.
  else {
    vector<fastjet::PseudoJet> constituents = theClusterSequence.constituents( jet );
    GreaterByEtPseudoJet compEt;
    sort( constituents.begin(), constituents.end(), compEt );
    return constituents[0];
  }
}


//-------------------------------------------------------------------------
// Find the calo tower associated with the jet
CaloTowerDetId CATopJetAlgorithm::getCaloTower( const fastjet::PseudoJet & jet,
						const CaloTowerCollection & fInput,
						const CaloSubdetectorGeometry  * fTowerGeometry,
						const fastjet::ClusterSequence & theClusterSequence ) const
{

  // If the jet is just a single calo tower, this is trivial
  bool isTower = jet.user_index() > 0;
  // This is where it really IS index 0.
  // There's a bug in fastjet. They set the user_index to 0 instead of -1 for the previous output. 
  // Need to check if it's "really" index 0 in input, or it's a reconstructed jet.
  if ( jet.user_index() == 0 && jet.perp() > 0 && jet.perp() == fInput[0].pt() ) isTower = true;

  if ( isTower ) {
    return fInput[jet.user_index()].id();
  }

  // If the user requested to get the max tower, return the max tower
  if ( useMaxTower_ ) {
    return fInput[getMaxTower( jet, fInput, theClusterSequence ).user_index()].id();
  }


  // Otherwise, we find the closest calorimetery tower to the jet centroid
  reco::Particle::LorentzVector v( jet.px(), jet.py(), jet.pz(), jet.e() );
  GlobalPoint centroid( v.x(), v.y(), v.z() );

  // Find the closest calo det id
  CaloTowerDetId detId ( fTowerGeometry->getClosestCell( centroid ) );

  // Return closest calo det id
  return detId;
}

//-------------------------------------------------------------------------
// Get number of calo towers away that the two calo towers are
int CATopJetAlgorithm::getDistance ( CaloTowerDetId const & t1, CaloTowerDetId const & t2, 
				     const CaloSubdetectorGeometry  * fTowerGeometry ) const
{

  int ieta1 = t1.ieta();
  int iphi1 = t1.iphi();
  int ieta2 = t2.ieta();
  int iphi2 = t2.iphi();

  int deta = abs(ieta2 - ieta1);
  int dphi = abs(iphi2 - iphi1);
  
  while ( dphi >= CaloTowerDetId::kMaxIEta ) dphi -= CaloTowerDetId::kMaxIEta;
  while ( dphi <= 0 ) dphi += CaloTowerDetId::kMaxIEta;

  return deta + dphi;
}

//-------------------------------------------------------------------------
// attempt to decompose a jet into "hard" subjets, where hardness is set by ptHard
//
bool CATopJetAlgorithm::decomposeJet(const fastjet::PseudoJet & theJet, 
				     const fastjet::ClusterSequence & theClusterSequence, 
				     const CaloTowerCollection & fInput,
				     const CaloSubdetectorGeometry  * fTowerGeometry,
				     double ptHard, int nCellMin,
				     fastjet::PseudoJet & ja, fastjet::PseudoJet & jb, 
				     vector<fastjet::PseudoJet> & leftovers) const {

  bool goodBreak;
  fastjet::PseudoJet j = theJet;
  leftovers.clear();
  
  while (1) {                                                      // watch out for infinite loop!
    goodBreak = theClusterSequence.has_parents(j,ja,jb);
    if (!goodBreak)                                 break;         // this is one cell, can't decluster anymore

    if ( useAdjacency_ &&
	 adjacentCells(ja,jb,fInput,
		       fTowerGeometry,
		       theClusterSequence,
		       nCellMin) )                  break;         // the clusters are "adjacent" in the calorimeter => shouldn't have decomposed
    if (ja.perp() < ptHard && jb.perp() < ptHard)   break;         // broke into two soft clusters, dead end
    if (ja.perp() > ptHard && jb.perp() > ptHard)   return true;   // broke into two hard clusters, we're done!
    else if (ja.perp() > jb.perp()) {                              // broke into one hard and one soft, ditch the soft one and try again
      j = ja;
      vector<fastjet::PseudoJet> particles = theClusterSequence.constituents(jb);
      leftovers.insert(leftovers.end(),particles.begin(),particles.end());
    }
    else {
      j = jb;
      vector<fastjet::PseudoJet> particles = theClusterSequence.constituents(ja);
      leftovers.insert(leftovers.end(),particles.begin(),particles.end());
    }
  }

  // did not decluster into hard subjets
  ja.reset(0,0,0,0);
  jb.reset(0,0,0,0);
  leftovers.clear();
  return false;
}
