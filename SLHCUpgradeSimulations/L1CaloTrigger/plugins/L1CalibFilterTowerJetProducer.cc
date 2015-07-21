// -*- C++ -*-
//
// Package:    L1CalibFilterTowerJetProducer
// Class:      L1CalibFilterTowerJetProducer
// 
/**\class CalibTowerJetCollection L1CalibFilterTowerJetProducer.cc MVACALIB/L1CalibFilterTowerJetProducer/src/L1CalibFilterTowerJetProducer.cc

 Description: Mk1 developer release

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Robyn Elizabeth Lucas,510 1-002,+41227673823,
//         Created:  Mon Nov 19 10:20:06 CET 2012
// $Id: L1CalibFilterTowerJetProducer.cc,v 1.7 2013/05/16 17:35:30 mbaber Exp $
//
// Modifications  :  Mark Baber Imperial College, London 


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
#include <string>


#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/LUT.h"


// Ranking function for sort
bool towerJetRankDescending ( l1slhc::L1TowerJet jet1, l1slhc::L1TowerJet jet2 ){      return ( jet1.p4().Pt() > jet2.p4().Pt() ); }
bool L1JetRankDescending ( l1extra::L1JetParticle jet1, l1extra::L1JetParticle jet2 ){ return ( jet1.p4().Pt() > jet2.p4().Pt() ); }





//
// class declaration
//


using namespace l1slhc;
using namespace edm;
using namespace std;
using namespace reco;
using namespace l1extra;



class L1CalibFilterTowerJetProducer : public edm::EDProducer {
   public:
      explicit L1CalibFilterTowerJetProducer(const edm::ParameterSet&);
      ~L1CalibFilterTowerJetProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      virtual l1extra::L1JetParticleCollection calibrateJetCollection( edm::Handle<L1TowerJetCollection> UncalibJet_Tower, LUT calibLUT );


      // ----------member data ---------------------------
      ParameterSet conf_;

      // Jet pt threshold for jet energies used in the calculation of Ht, mHt
      double energySumsJetPtThreshold;

      //  L1 pT calibration threshold, minimum L1 jet pT to apply correction  
      double pTCalibrationThreshold;
      // Boundaries of eta segmentation for calibration
      std::vector <double> etaRegionSlice;

      // Number of parameters used in calibration
      uint calibParameters;

      // Calibration look up table 
      LUT calibrationLUT;
      TString lutFilename;

};


L1CalibFilterTowerJetProducer::L1CalibFilterTowerJetProducer(const edm::ParameterSet& iConfig):
conf_(iConfig)
{
    produces<L1JetParticleCollection>( "UncalibratedTowerJets" ) ;
    produces<L1JetParticleCollection>( "CalibratedTowerJets"   ) ;
    produces<L1EtMissParticleCollection>( "TowerMHT" ) ;

    // Read the calibration eta segementation
    etaRegionSlice        = iConfig.getParameter< std::vector<double> >("EtaRegionSlice");
    sort (etaRegionSlice.begin(), etaRegionSlice.end());  // ensure the bins are in ascending order  


    edm::FileInPath LUTFile = iConfig.getParameter<edm::FileInPath>("CalibrationLUTFile");
    calibParameters         = iConfig.getParameter< uint >("CalibrationParameters");

    // Calculate LUT dimensions and initialise LUT
    int rows    = int(calibParameters);
    int columns = etaRegionSlice.size() - 1;
    calibrationLUT = LUT( rows, columns );

    // Load the look up table
    calibrationLUT.loadLUT( LUTFile.fullPath().c_str() );

    // Extract jet pT threshold for Ht, mHt calculation
    energySumsJetPtThreshold   = iConfig.getParameter<double> ("EnergySumsJetPtThreshold");

    // Minimum PT for calibration. Remove jets which cannot be calibrated
    pTCalibrationThreshold     = iConfig.getParameter<double> ("pTCalibrationThreshold");
   

}


L1CalibFilterTowerJetProducer::~L1CalibFilterTowerJetProducer()
{
}

void
L1CalibFilterTowerJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
   bool evValid = true;

   auto_ptr<L1JetParticleCollection> outputCalibratedL1Extra(new L1JetParticleCollection());
   auto_ptr<L1JetParticleCollection> outputUncalibratedL1Extra(new L1JetParticleCollection());


   auto_ptr<l1extra::L1EtMissParticleCollection> outputmht(new L1EtMissParticleCollection());
   auto_ptr<double> outHT   (new double());


   // Read in collection depending on parameters in SLHCCaloTrigger_cfi.py
   edm::Handle<L1TowerJetCollection > UncalibratedTowerJets;
   iEvent.getByLabel(conf_.getParameter<edm::InputTag>("UncalibratedTowerJets"), UncalibratedTowerJets);
   if(!UncalibratedTowerJets.isValid()){
     edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("UncalibratedTowerJets") << std::endl; 
     evValid = false;
   }


   if( evValid ) {
    
     double ht(0);
     math::PtEtaPhiMLorentzVector mht, tempJet;
 
     // Perform jet eta-pT dependent pT calibrations
     l1extra::L1JetParticleCollection calibJets = calibrateJetCollection( UncalibratedTowerJets, calibrationLUT );

     // Store calibrated jets and perform calibrated jet energy sums
     for ( l1extra::L1JetParticleCollection::const_iterator calib_Itr = calibJets.begin(); calib_Itr!= calibJets.end() ; ++calib_Itr ){

       L1JetParticle calibJet = (*calib_Itr);
       outputCalibratedL1Extra->push_back( calibJet );

       // Add jet to energy sums if energy is above threshold 
       if( calibJet.p4().pt() > energySumsJetPtThreshold ){

	 tempJet.SetCoordinates( calibJet.p4().pt(), calibJet.p4().eta(), calibJet.p4().phi(), calibJet.p4().M() );
         ht  += tempJet.pt();
         mht += tempJet;

       }

     }


  
     // Store uncalibrated jet collection
     for ( L1TowerJetCollection::const_iterator Uncalib_Itr = UncalibratedTowerJets->begin(); Uncalib_Itr != UncalibratedTowerJets->end(); ++Uncalib_Itr ){
    
       L1TowerJet uncalibJet = (*Uncalib_Itr);

       math::PtEtaPhiMLorentzVector tempJet;
       tempJet.SetCoordinates( uncalibJet.p4().pt(), uncalibJet.p4().eta(), uncalibJet.p4().phi(), uncalibJet.p4().M() );

       outputUncalibratedL1Extra->push_back( l1extra::L1JetParticle( tempJet, l1extra::L1JetParticle::JetType::kCentral, 0 ) );
     }
     

     // create L1Extra object
     math::PtEtaPhiMLorentzVector p4tmp = math::PtEtaPhiMLorentzVector( mht.pt(), 0., mht.phi(), 0. );
	
     L1EtMissParticle l1extraMHT(p4tmp, L1EtMissParticle::kMHT, ht,
				 Ref< L1GctEtMissCollection >(), Ref< L1GctEtTotalCollection >(),
				 Ref< L1GctHtMissCollection >(), Ref< L1GctEtHadCollection >()  ,  0);
     
     outputmht->push_back(l1extraMHT);

     iEvent.put( outputUncalibratedL1Extra,"UncalibratedTowerJets");
     iEvent.put( outputCalibratedL1Extra,  "CalibratedTowerJets");


     iEvent.put( outputmht,                "TowerMHT" );
     
     
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CalibFilterTowerJetProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CalibFilterTowerJetProducer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
L1CalibFilterTowerJetProducer::beginRun(edm::Run&, edm::EventSetup const&)
{    
}

// ------------ method called when ending the processing of a run  ------------
void 
L1CalibFilterTowerJetProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
L1CalibFilterTowerJetProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
L1CalibFilterTowerJetProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1CalibFilterTowerJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//
// member functions
//




// Calibrates an input TowerJet collection with the specified input look up table. The resulting calibrated jets are sorted in pT 
// and are returned as a L1JetParticleCollection.
l1extra::L1JetParticleCollection
L1CalibFilterTowerJetProducer::calibrateJetCollection( edm::Handle<L1TowerJetCollection> UncalibJet_Tower, LUT calibLUT ){



     std::vector <L1JetParticle> unsortedCalibratedL1Jets;
  
     // Calibrate jets with LUT
     for ( L1TowerJetCollection::const_iterator Uncalib_It = UncalibJet_Tower->begin(); Uncalib_It != UncalibJet_Tower->end(); ++Uncalib_It ){
    
       L1TowerJet uncalibJet = (*Uncalib_It);
 
       // Jet pT threshold for calibration, only calibrate above threshold
       if ( uncalibJet.p4().pt() < pTCalibrationThreshold ){

	 // Uncomment to Store un-calibrated L1Jet 
	 // ------------------------------------------------------------                                                                                        
	 // Create un-sorted calibrated L1Jet 
	 //math::PtEtaPhiMLorentzVector tempJet;
	 //tempJet.SetCoordinates( uncalibJet.p4().pt(), uncalibJet.p4().eta(), uncalibJet.p4().phi(), uncalibJet.p4().M() );
	 //unsortedCalibratedL1Jets.push_back( l1extra::L1JetParticle( tempJet, l1extra::L1JetParticle::JetType::kCentral, 0 ) );

	 continue;
       }
       
       // Find in which Eta bin the jet resides 
       uint pEta;
       for (pEta = 1; pEta < etaRegionSlice.size(); ++pEta){

	 double eta = uncalibJet.p4().eta();

	 // Get Eta bin lower and upper bounds 
	 double EtaLow  = etaRegionSlice[ pEta - 1];
	 double EtaHigh = etaRegionSlice[ pEta ];
	 // Eta resides within current boundary
	 if ( (eta >= EtaLow) && (eta < EtaHigh) ){
	   break; // found the correct eta bin, break
	 }

       }

       // Extract eta dependent correction factors
       // **************************************************
       int etaIndex       = pEta - 1;// Correct for array starting at zero


       // Perform jet calibration
       // ------------------------------------------------------------
       double unCorrectedPt          = uncalibJet.p4().pt(); 
       double correctedPt            = 0;

	 
       // Get calibration parameters for given eta bin
       double p0 = calibLUT.getElement( etaIndex, 0 );
       double p1 = calibLUT.getElement( etaIndex, 1 );
       double p2 = calibLUT.getElement( etaIndex, 2 );
       double p3 = calibLUT.getElement( etaIndex, 3 );
       double p4 = calibLUT.getElement( etaIndex, 4 );
       double p5 = calibLUT.getElement( etaIndex, 5 );



       // Calucualte the jet correction
       double logPt = log( unCorrectedPt );
       
       double term1 = p1 / ( logPt * logPt + p2 );
       double term2 = p3 * exp( -p4*((logPt - p5)*(logPt - p5)) );
       
       // Calculate the corrected Pt 
       double correction    = (p0 + term1 + term2);
       //double invCorrection = 1/correction;
       
       // Use invereted correction for Phase 2
       //correctedPt    = invCorrection*unCorrectedPt;
       	 correctedPt    = correction*unCorrectedPt;

       



       
       // Create and store unsorted calibrated L1Jet collection
       // ------------------------------------------------------------
       
       // Create un-sorted calibrated L1Jet 
       math::PtEtaPhiMLorentzVector tempJet;
       tempJet.SetCoordinates( correctedPt, uncalibJet.p4().eta(), uncalibJet.p4().phi(), uncalibJet.p4().M() );
       unsortedCalibratedL1Jets.push_back( l1extra::L1JetParticle( tempJet, l1extra::L1JetParticle::JetType::kCentral, 0 ) );
       
     }     
     
     // Sort calibrated jets
     std::sort( unsortedCalibratedL1Jets.begin(),         unsortedCalibratedL1Jets.end(),         L1JetRankDescending );
     
     // Return the sorted, calibrated jets
     return unsortedCalibratedL1Jets;
    

}














//define this as a plug-in
DEFINE_FWK_MODULE(L1CalibFilterTowerJetProducer);
