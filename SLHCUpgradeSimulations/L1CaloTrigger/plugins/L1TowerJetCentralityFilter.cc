// Original Author:  Mark Baber Imperial College, London



// system include files
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TLorentzVector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include <iostream>
#include <fstream>

#include "FWCore/ParameterSet/interface/FileInPath.h"

// Bitonic sorting functions
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/BitonicSort.hpp"

#include <algorithm>

//
// class declaration
//


using namespace l1slhc;
using namespace edm;
using namespace std;
using namespace reco;
using namespace l1extra;





struct tempJet
{

  tempJet( ) :
      mJet( NULL )
  {}        //constructor


  tempJet( const l1slhc::L1TowerJet& aJet ) :
      mJet( &aJet )
  { /*....*/	}        //constructor

  const   l1slhc::L1TowerJet* mJet;
};


/*
bool operator> ( tempJet& aA , tempJet& aB )
{

  if( aA.mJet == NULL ) return false;	//Jets that don't exist: require 128 or 64 spots
  if( aB.mJet == NULL ) return true;

  if ( aA.mJet->E() > aB.mJet->E() ) return true;
  if ( aA.mJet->E() < aB.mJet->E() ) return false;


  //those aA and aB with the same energy are all that remain
  // if ( *(aA.mComparisonDirection) == phi ){
    return ( abs( aA.mJet-> AsymPhi() ) <= abs( aB.mJet->AsymPhi() ) );
    //  }else{
    return ( abs( aA.mJet-> AsymEta() ) <= abs( aB.mJet->AsymEta() ) );	
    //  }	

}
*/

  
bool operator>( tempJet & aLeft, tempJet & aRight ){
  //	std::cout << "works???\n";


//   std::cout << "Pt1 = "     << aLeft.mJet->Pt()         << "\tPt2 = "   << aRight.mJet->Pt()
// 	    << "\tiEta1 = " << aLeft.mJet->iEta()       << "\tiPhi1 = " << aLeft.mJet->iPhi()      
// 	    << "\tiEta2 = " << aRight.mJet->iEta() 	<< "\tiPhi2 = " << aRight.mJet->iPhi()
// 	    << "\tCent1 = " << aLeft.mJet->Centrality() << "\tCent2 = " << aRight.mJet->Centrality();


  if ( aLeft.mJet->Pt() == aRight.mJet->Pt() ){

    // Degenerate energies
    // ~~~~~~~~~~~~~~~~~~~
    // Order by the lowest centrality, for degenerate centralities favour the right jet to avoid needless sorting.
    // Result: Favours jets with the lowest iEta and iPhi, due to the original jet ordering
    
    //      std::cout << "\t1>2 = " << ( aLeft.mJet->Centrality() < aRight.mJet->Centrality() ) << "\n";
    return ( aLeft.mJet->Centrality() < aRight.mJet->Centrality() );
    
  }
  else{

    // Rank by pT
    // ~~~~~~~~~~
    //    std::cout << "\t1>2 = " << ( aLeft.mJet->Pt() > aRight.mJet->Pt() ) << " ENERGIES NOT DEGENERATE\n";
    return ( aLeft.mJet->Pt() > aRight.mJet->Pt() );
  }


}

// Sort with srd::sort
bool tempJetSort (tempJet aJet, tempJet bJet) { return ( aJet>bJet); }
//<TEMP>






class L1TowerJetCentralityFilter : public edm::EDProducer {
   public:
      explicit L1TowerJetCentralityFilter(const edm::ParameterSet&);
      ~L1TowerJetCentralityFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
    
       // ----------member data ---------------------------
      ParameterSet conf_;
    
      // Limit of the number of jets to be retained in filtering
      int mNumOfOutputJets;  

      std::auto_ptr < l1slhc::L1TowerJetCollection > mOutputCollection;


};

//
// constructors and destructor
//

L1TowerJetCentralityFilter::L1TowerJetCentralityFilter(const edm::ParameterSet& iConfig):
  conf_(iConfig),
  mNumOfOutputJets( iConfig.getParameter<uint32_t>("NumOfOutputJets") )
{
  //    produces< l1slhc::L1TowerJetCollection >("FilteredTowerJets");
    produces< l1slhc::L1TowerJetCollection >( );
    std::cout << "\n\n----------------------------------------\nBegin: L1TowerJetCentralityFilter\n----------------------------------------\n\n";

}

L1TowerJetCentralityFilter::~L1TowerJetCentralityFilter()
{
}




// ------------ method called to produce the data  ------------
void
L1TowerJetCentralityFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
    bool evValid = true;

    edm::Handle< L1TowerJetCollection > preFiltJets;
    iEvent.getByLabel(conf_.getParameter<edm::InputTag>("PreFilteredJets"), preFiltJets);
    if(!preFiltJets.isValid()){
        evValid=false;
    }

    if( !evValid ) {
        throw cms::Exception("MissingProduct") << conf_.getParameter<edm::InputTag>("preFiltJets")      
	  				       << std::endl; 
    }
    else{

      auto_ptr< L1TowerJetCollection > filteredJets(new L1TowerJetCollection()); // Overlap filtered jets



      //      std::cout << "FLAG\n\n";

      //      std::vector < const l1slhc::L1TowerJet* > lTowerVector;
      std::vector < tempJet > tempJetVec;

  

      for (L1TowerJetCollection::const_iterator jetIt = preFiltJets->begin(); jetIt!= preFiltJets->end(); ++jetIt ){
	
	L1TowerJet h = *jetIt;

	tempJetVec.push_back(  tempJet( *jetIt ) );
	
	// FUNKY, SORT THIS OUT...
  //	lTowerVector.push_back( &(*jetIt) );
	
	//	lTowerVector.push_back( *jetIt );
		
      }





    
      
      // Resize array to a power of 2 for the Bitonic sort algorithm
      // Find the smallest power of 2 that contains the require jet multiplicity


    // PERHAPS CALCULATE THE NEAREST POWER OF TWO, THIS IS OVERKILL. I.E BIT SHIFT LEFT AND "AND" THE OUTPUT TO GET MSB
      //      lTowerVector.resize(4096);

//       for (unsigned int i = 0; ((i < lTowerVector.size()) && (i < 100)); i++){
// 	if ( lTowerVector[i] != NULL ){
// 	  std::cout << "ET = "           << lTowerVector[i]->Pt()   << "\tCentrality = " << lTowerVector[i]->Centrality() 
// 		    << i << "\tiEta = "  << lTowerVector[i]->iEta() << "\tiPhi = "       << lTowerVector[i]->iPhi() 
// 		    << "\n";
// 	}
//       }

      //      lJetWrapper2DVector.resize(4096); //resize to nearest power of 2 to 72*56 for the bitonic sort algorithm
      



    /*
    // Fill the rest with NULL jets, INCREDIBLE INEFFICIENCT, USE THE RESIZE ROUTINE!!!!!!!!!!!!!!!!!
    for (int i = lTowerVector.size(); i <= 4096; i++){

      l1slhc::L1TowerJet* mJet(NULL);

      lTowerVector.push_back( mJet );
    }

    std::cout << lTowerVector.size() << "\n\n\n";


    // PERHAPS CALCULATE THE NEAREST POWER OF TWO, THIS IS OVERKILL. I.E BIT SHIFT LEFT AND "AND" THE OUTPUT TO GET MSB

    lTowerVector.resize(4096); //resize to nearest power of 2 to 72*56 for the bitonic sort algorithm
    */


      // **********************************************************************
      // *                             Rank jets                              *
      // **********************************************************************
      //
      // Rank jets in descending Et, in the case of degenerate energies, rank by the 
      // 'centrality' secondary sort parameter.
      // ---------------------------------------------------------------------

      // sort jets around eta/phi by energy and relevant asymmetry parameter
//       L1TowerJetCollection::iterator lStart( preFiltJets->begin() );
//       L1TowerJetCollection::iterator   lEnd( preFiltJets->end() );	

//       std::cout << "Prior to start\n";
//       //      std::vector< L1TowerJet >::iterator lStart( filteredJets->begin() );
//       std::vector< const l1slhc::L1TowerJet* >::iterator lStart( lTowerVector.begin() );
//       std::cout << "Prior to End\n";
//       //      std::vector< L1TowerJet >::iterator   lEnd( filteredJets->end() );
//       std::vector< const l1slhc::L1TowerJet* >::iterator   lEnd( lTowerVector.end() );
//       std::cout << "Prior to sort\n";
//       BitonicSort< const l1slhc::L1TowerJet* >( down , lStart , lEnd );
//       std::cout << "After sort\n";


/*
      for (unsigned int i = 0; ((i < tempJetVec.size()) && (i < 100)); i++){
	if ( tempJetVec[i].mJet != NULL ){
	  std::cout << i << "\tET = "     << tempJetVec[i].mJet->Pt()   << "\tCentrality = " << tempJetVec[i].mJet->Centrality() 
		    << "\tiEta = "        << tempJetVec[i].mJet->iEta() << "\tiPhi = "       << tempJetVec[i].mJet->iPhi() 
		    << "\n\tET = "     << tempJetVec[i+1].mJet->Pt()   << "\tCentrality = " << tempJetVec[i+1].mJet->Centrality() 
		    << "\tiEta = "        << tempJetVec[i+1].mJet->iEta() << "\tiPhi = "       << tempJetVec[i+1].mJet->iPhi() 
		    << "\n\tJet1 > Jet2 = " << ( tempJetVec[i] > tempJetVec[i+1] )
		    << "\n\n";
	}
      }
*/

      /*
  std::cout << "Prior to start\n";
//      std::vector< L1TowerJet >::iterator lStart( filteredJets->begin() );
 std::vector< tempJet >::iterator lStart2( tempJetVec.begin() );
 std::cout << "Prior to End\n";
 //      std::vector< L1TowerJet >::iterator   lEnd( filteredJets->end() );
 std::vector< tempJet >::iterator   lEnd2( tempJetVec.end() );
 std::cout << "Prior to sort\n";
 // BitonicSort< tempJet >( down , lStart2 , lEnd2 );
 BitonicSort< tempJet >( down , lStart2 , lEnd2 );
 std::cout << "After sort\n";
      */



      // Sort with the std library
      std::sort (tempJetVec.begin(), tempJetVec.end(), tempJetSort);


//       for (unsigned int i = 0; ((i < lTowerVector.size()) && (i < 100)); i++){
// 	if ( lTowerVector[i] != NULL ){
// 	  std::cout << "ET = "           << lTowerVector[i]->Pt()   << "\tCentrality = " << lTowerVector[i]->Centrality() 
// 		    << i << "\tiEta = "  << lTowerVector[i]->iEta() << "\tiPhi = "       << lTowerVector[i]->iPhi() 
// 		    << "\n";
// 	  std::cout << (lTowerVector[i] < lTowerVector[i+1])  << "\n";
// 	  //	  << *lTowerVector[i]
// 	}
//       }



/*
      for (unsigned int i = 0; ((i < tempJetVec.size()) && (i < 100)); i++){
	if ( tempJetVec[i].mJet != NULL ){
	  std::cout << i << "\tET = "     << tempJetVec[i].mJet->Pt()   << "\tCentrality = " << tempJetVec[i].mJet->Centrality() 
		    << "\tiEta = "        << tempJetVec[i].mJet->iEta() << "\tiPhi = "       << tempJetVec[i].mJet->iPhi() 
		    << "\n\tET = "     << tempJetVec[i+1].mJet->Pt()   << "\tCentrality = " << tempJetVec[i+1].mJet->Centrality() 
		    << "\tiEta = "        << tempJetVec[i+1].mJet->iEta() << "\tiPhi = "       << tempJetVec[i+1].mJet->iPhi() 
		    << "\n\tJet1 > Jet2 = " << ( tempJetVec[i] > tempJetVec[i+1] )
		    << "\n\n";
 
	}
      }
*/




//      std::vector<JetWrapper2D>::iterator lStart( lJetWrapper2DVector.begin() );
//      std::vector<JetWrapper2D>::iterator   lEnd( lJetWrapper2DVector.end()   );	
//      BitonicSort< JetWrapper2D >( down , lStart , lEnd );



      // iEta, iPhi vetoed positions
      std::deque< std::pair<int, int> > lVetos; 
      // Number of jets currently retained
      int lJetCounter(0);

      //THESE AREN'T FILTERED YET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
            

      //      for (L1TowerJetCollection::const_iterator jetIt = preFiltJets->begin(); jetIt!= preFiltJets->end(); ++jetIt ){
      //      for( std::vector<L1TowerJet>::iterator jetIt = lTowerVector.begin(); jetIt != lTowerVector.end(); ++jetIt){
      for (unsigned int iTemp = 0; iTemp < tempJetVec.size(); iTemp++){
      //for( std::vector<JetWrapper2D>::iterator lIt =lJetWrapper2DVector.begin(); lIt != lJetWrapper2DVector.end(); ++lIt){

     
	// NOTE: Erroneous vetoting of non-square jet shapes still remains. Problem can only 
	//       be resolved if mJetShapeMap of L1TowerJetProducer becomes a member variable
	//       of L1TowerJet OR if we pass the jet map from module to module 
	// ##########################!!!!DO THE LATTER!!!!##################################################

	// **********************************************************************
	// *                            Jet filtering                           *
	// **********************************************************************
	//
	// Pick the highest ranked jets for output and veto all the jets that overlap its position.
	// -----------------------------------------------------------------------
	
	//	if( tempJetVec[iTemp].mJet ){ //if jet exists	
	  int lJetsize =  tempJetVec[iTemp].mJet->JetSize() ;
	  bool lVetoed( false );
  
	  // Check jet iEta and iPhi against list of vetod iEta and iPhi. If jet is vetoed skip to next jet in sorted list.
	  for( std::deque< std::pair<int, int> >::iterator vetoIt = lVetos.begin() ; vetoIt != lVetos.end() ; ++vetoIt ){
  

	    if ( ( tempJetVec[iTemp].mJet->iEta() == vetoIt->first ) && ( tempJetVec[iTemp].mJet->iPhi() == vetoIt->second) ){ 
	      // Jet is already vetoed break
	      
	      //	      std::cout << "Jet with iEta,iPhi = " << tempJetVec[iTemp].mJet->iEta() << "," << tempJetVec[iTemp].mJet->iPhi() << " vetoed.\n";
	      lVetoed = true;   break;
	    }

	  }
 
	  if( !lVetoed ){	// If the jet is not vetoed then add to the output collection and add veto region to veto list.


	    // Store the overlap filtered jet

// 	    std::cout << "Pt = " << tempJetVec[iTemp].mJet->p4().Pt()     << "\t" 
// 		      << tempJetVec[iTemp].mJet->WeightedEta() << "\t" 
// 		      << tempJetVec[iTemp].mJet->WeightedPhi() << "\n"

	    double weightedEta =  tempJetVec[iTemp].mJet->WeightedEta();
	    double weightedPhi =  tempJetVec[iTemp].mJet->WeightedPhi();
	    filteredJets->insert( weightedEta, weightedPhi, *tempJetVec[iTemp].mJet );

	    //	    filteredJets->insert( jetIt->iEta() , jetIt->iPhi() , (*jetIt) );
	    lJetCounter++;

	    // Generate the veto list
	    for( int i = -lJetsize +1 ; i != lJetsize ; ++i ){
  

	      int lPhi( tempJetVec[iTemp].mJet->iPhi() + i );


	      if( lPhi > 72 ) lPhi -= 72;
	      if( lPhi < 1 )  lPhi += 72;
	      // Single veto coordinate (iEta, iPhi)
	      std::pair<int,int> veto;
	      veto.second = lPhi;
	      // For each iPhi veto all overlapping iEta positions
	      for( int j = -lJetsize +1 ; j != lJetsize ; ++j ){

		int lEta( tempJetVec[iTemp].mJet->iEta() + j );
  
		// No iEta = 0 in this coordinate system: need to allow for this
		//    Accounts for the range scanned being too small, due to no
		//    iEta = 0 being present, by jumping to the end of the range.
		//    Obtuse but effective.
		if( (lEta == 0) && (j < 0) ) lEta = tempJetVec[iTemp].mJet->iEta() - lJetsize;
		if( (lEta == 0) && (j > 0) ) lEta = tempJetVec[iTemp].mJet->iEta() + lJetsize;
		veto.first = lEta;

		// Add vetoed coordinate to list of vetoes
		lVetos.push_back( veto );
		//		std::cout << "(" << lEta << ", " << lPhi << ") vetoed.\n";
	      }
	    }
	    if( lJetCounter >= mNumOfOutputJets ) break;  // Restrict number of jets retained in filtering
	  }

// 	}
// 	else{
// 	  std::cout << "This actually does something\n";
// 	}
      

      } // End jet loop


      // Store the overlap filtered jets
      //      iEvent.put( filteredJets,"FilteredTowerJets");
      iEvent.put( filteredJets );

    } // End valid event


}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TowerJetCentralityFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TowerJetCentralityFilter::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
L1TowerJetCentralityFilter::beginRun(edm::Run&, edm::EventSetup const&)
{    
}

// ------------ method called when ending the processing of a run  ------------
void 
L1TowerJetCentralityFilter::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
L1TowerJetCentralityFilter::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
L1TowerJetCentralityFilter::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TowerJetCentralityFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TowerJetCentralityFilter);
