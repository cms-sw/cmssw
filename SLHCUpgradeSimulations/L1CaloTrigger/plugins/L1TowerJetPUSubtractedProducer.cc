//This producer creates a PU subtracted collection of upgrade jets in both slhc::L1TowerJet format
//and l1extra::L1JetParticle format
//
//It takes rho produced by L1TowerJetPUSubtraction producer and applies PU subtraction to input collection which
//is filtered, uncalibrated jet collection
//
//Calibration should be done after this step is completed
//
// Original Author:  Robyn Elizabeth Lucas,510 1-002,+41227673823,
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

//
// class declaration
//


using namespace l1slhc;
using namespace edm;
using namespace std;
using namespace reco;
using namespace l1extra;


class L1TowerJetPUSubtractedProducer : public edm::EDProducer {
   public:
      explicit L1TowerJetPUSubtractedProducer(const edm::ParameterSet&);
      ~L1TowerJetPUSubtractedProducer();

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

      // Jet pt threshold for jet energies to be retained after PU subtraction
      double jetPtPUSubThreshold;

      // Local rho eta region boundaries
      vector < double > localRhoEtaDiv;


};

L1TowerJetPUSubtractedProducer::L1TowerJetPUSubtractedProducer(const edm::ParameterSet& iConfig):
conf_(iConfig)
{

  std::cout << "\n\n----------------------------------------\nBegin: L1TowerJetPUSubtractedProducer\n----------------------------------------\n\n";
    produces<L1TowerJetCollection>("PrePUSubCenJets");
    produces<L1TowerJetCollection>("PUSubCenJets");    
    produces<L1TowerJetCollection>("LocalPUSubCenJets");

    produces<L1JetParticleCollection>("PUSubCen8x8") ;
    produces<L1TowerJetCollection>("CalibFwdJets");

    // Extract pT threshold for retaining PU subtracted jets
    jetPtPUSubThreshold = iConfig.getParameter<double> ("JetPtPUSubThreshold");


    // Divisions of the eta regions overwhich to calculate local rhos
    localRhoEtaDiv  = iConfig.getParameter< vector< double > >("LocalRhoEtaDivisions");
    // Ensure the divisions are in ascending order of eta
    sort (localRhoEtaDiv.begin(), localRhoEtaDiv.end());


}


L1TowerJetPUSubtractedProducer::~L1TowerJetPUSubtractedProducer()
{
}




// ------------ method called to produce the data  ------------
void
L1TowerJetPUSubtractedProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   
    bool evValid = true;
    auto_ptr< L1TowerJetCollection > outputCollCen(new L1TowerJetCollection());         // Global PU-subtracted central jets
    auto_ptr< L1TowerJetCollection > cenLocalPUS(new L1TowerJetCollection());           // Local  PU-subtracted central jets
    auto_ptr< L1TowerJetCollection > outputCollCenPrePUSub(new L1TowerJetCollection()); // Pre PU-subtracted central jets
    auto_ptr< L1JetParticleCollection > outputExtraCen(new L1JetParticleCollection());


    // WARNING: May or may not be calibrated, depending on the configuration parameters given for L1TowerJetPUEstimator
    edm::Handle< double > calRho;
    iEvent.getByLabel(conf_.getParameter<edm::InputTag>("CalibratedL1Rho"), calRho);
    if(!calRho.isValid()){
      evValid=false;
      edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("CalibratedL1Rho") << std::endl; 
    }

    edm::Handle<L1TowerJetCollection > UnCalibCen;
    iEvent.getByLabel(conf_.getParameter<edm::InputTag>("FilteredCircle8"), UnCalibCen);
    if(!UnCalibCen.isValid()){
      edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("FilteredCircle8") << std::endl; 
      evValid=false;
    }


        
    // Local PU subtraction
    // ******************************
    edm::Handle< vector <double> > LocalRho;
    iEvent.getByLabel(conf_.getParameter<edm::InputTag>("LocalRho"), LocalRho);
    if(!LocalRho.isValid()){
      edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("LocalRho") << std::endl;
      evValid=false;
    }
    /*    
    edm::Handle< vector <double> > LocalRhoBoundaries;
    iEvent.getByLabel(conf_.getParameter<edm::InputTag>("LocalRhoBoundaries"), LocalRhoBoundaries);
    if(!LocalRhoBoundaries.isValid()){
      edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("LocalRhoBoundaries") << std::endl;
      evValid=false;
    }
    */


    if( evValid ) {

      // Get rho from the producer L1TowerJetPUSubtraction
      // This value is calibrated to offline calo rho if useRhoCalibration in the config file is set to true
      double cal_rhoL1 = *calRho;


      // Retrive local rhos and eta boundries corresponding to these rhos
      // Local PU subtraction
      vector <double> lRho           = *LocalRho;
      //      vector <double> localRhoEtaDiv = localRhoEtaDiv;//*LocalRhoBoundaries;
      

      ///////////////////////////////////////////////////
      //              JET VALUES 
      ///////////////////////////////////////////////////     
      
      math::PtEtaPhiMLorentzVector upgrade_jet;
    
      //Produce calibrated pt collection: central jets
      for (L1TowerJetCollection::const_iterator il1 = UnCalibCen->begin(); il1!= UnCalibCen->end(); ++il1 ){

          L1TowerJet h = (*il1);

	  // Extract the old tower jet information and store in a new tower jet
	  // Extremely awkward, to be fixed later
	  //	  double l1Pt_   = il1->Pt();
	  double unSubPt  = h.Pt();
          double weightedEta = il1->WeightedEta();
          double l1wPhi_ = il1->WeightedPhi() ;

	  // ****************************************
	  // *   Store the pre PU subtracted jets   *
	  // ****************************************
	  
	  math::PtEtaPhiMLorentzVector p4;
	  //use weighted eta and phi: these are energy weighted
	  p4.SetCoordinates(unSubPt , weightedEta , l1wPhi_ , il1->p4().M() );
	  h.setP4(p4);
	  
	  outputCollCenPrePUSub->insert( weightedEta , l1wPhi_ , h );

          //This is just for 8x8 circular jets: change if using different jets
          //double areaPerJet = 52 * (0.087 * 0.087) ;
          
	  // Get the eta*phi area of the jet
	  double areaPerJet = il1->JetRealArea();

	  // Perform the PU subtraction
          float l1Pt_PUsub_ = unSubPt - (cal_rhoL1 * areaPerJet);
	  
	  
	  // store the PU subtracted jets with Pt greater than specified threshold
          if(l1Pt_PUsub_ > jetPtPUSubThreshold){

            math::PtEtaPhiMLorentzVector p4;

            //use weighted eta and phi: these are energy weighted 
            p4.SetCoordinates(l1Pt_PUsub_ , weightedEta , l1wPhi_ , il1->p4().M() );
            h.setP4(p4);
          
	    // Store the PU subtracted towerjet
	    outputCollCen->insert( weightedEta , l1wPhi_ , h );

            upgrade_jet.SetCoordinates( l1Pt_PUsub_ , weightedEta , l1wPhi_ , il1->p4().M() );

            // add jet to L1Extra list
            outputExtraCen->push_back( L1JetParticle( math::PtEtaPhiMLorentzVector( l1Pt_PUsub_, weightedEta, l1wPhi_, 0. ),
						      Ref< L1GctJetCandCollection >(), 0 ) );
          }


	  // 
	  // Local PU subtraction
	  // 

	  double localSubtractedPt;

	  for (unsigned int iSlice = 0;iSlice < localRhoEtaDiv.size() - 1; iSlice++){

            // Get the current eta slice range
            double etaLow  = localRhoEtaDiv[iSlice];
            double etaHigh = localRhoEtaDiv[iSlice + 1];


            // Store the jet in the respective eta region
            if ( (weightedEta >= etaLow) && (weightedEta < etaHigh) ){


	      // CHECK THIS CORRESPONDS TO THE CORRECT REGION
	      double localRho = lRho[iSlice];


	      // Calculate the local PU subtrated pT
	      localSubtractedPt = unSubPt - localRho * areaPerJet;


	      //	      std::cout << "pT = " << unSubPt << "\tArea = " << areaPerJet << "\tEta = " << weightedEta << "\tetaRange = (" << etaLow << ", " << etaHigh 
	      //			<< ")\tPUS pT = " <<  localSubtractedPt<< "\n";
	      //	      std::cout << "\nEtaLow = " << etaLow << "\tEtaHigh = " << etaHigh << "\tLocalRho = " << localRho << "\tAreaPerJet = " 
	      //			<< areaPerJet <<  "\tUnPUSPt = " << unSubPt << "\tLocalPUSPt = " << localSubtractedPt << "\n";
	      	    
            }


          }





	  // store the PU subtracted jets with pT greater than specified threshold
          if(localSubtractedPt > jetPtPUSubThreshold){

	    // Local PUS jet
	    h.setPt(localSubtractedPt);

	    // Store the local PUS jet
	    cenLocalPUS->insert( weightedEta , l1wPhi_, h );

	  }

      }
    
      // Store the pre PU-subtracted, positive energy central jets
      iEvent.put(outputCollCenPrePUSub,"PrePUSubCenJets");

      //this is the slhc collection containing extra information
      iEvent.put(outputCollCen,"PUSubCenJets");

      iEvent.put(cenLocalPUS,"LocalPUSubCenJets");

      //this is the l1extra collection containing the same jet vector as in slhc collection
      iEvent.put(outputExtraCen,"PUSubCen8x8");

    }

}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TowerJetPUSubtractedProducer::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TowerJetPUSubtractedProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
L1TowerJetPUSubtractedProducer::beginRun(edm::Run&, edm::EventSetup const&)
{    

}

// ------------ method called when ending the processing of a run  ------------
void 
L1TowerJetPUSubtractedProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
L1TowerJetPUSubtractedProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
L1TowerJetPUSubtractedProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TowerJetPUSubtractedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//
// member functions
//

//define this as a plug-in
DEFINE_FWK_MODULE(L1TowerJetPUSubtractedProducer);
