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
// $Id: L1CalibFilterTowerJetProducer.cc,v 1.5 2013/03/21 16:11:18 rlucas Exp $
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

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

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
    
      float get_rho(double L1rho);

  //fwd calibration: V ROUGH (only to L1extra particles)
  //       double rough_ptcal(double pt);
    
      void TMVA_calibration();
      // ----------member data ---------------------------
      ParameterSet conf_;
    
      ifstream indata;

    
      edm::FileInPath inMVAweights_edm;
      float l1Pt, l1Eta;
      TMVA::Reader *reader;
      float val_pt_cal(float l1pt, float l1eta);

      // Jet pt threshold for jet energies used in the calculation of Ht, mHt
      double jetPtThreshold;
      // Parameter determining whether to perform L1 jet pt calibration to offline ak5 jets
      bool usePtCalib;
};


L1CalibFilterTowerJetProducer::L1CalibFilterTowerJetProducer(const edm::ParameterSet& iConfig):
conf_(iConfig)
{

    produces<L1TowerJetCollection>("CenJets");
//     produces<L1TowerJetCollection>("CalibFwdJets");
    produces<L1JetParticleCollection>( "Cen8x8" ) ;
//    produces< L1JetParticleCollection >( "Fwd8x8" ) ;
    produces<L1EtMissParticleCollection>( "TowerMHT" ) ;
    produces<double>("TowerHT");
    produces<bool>("JetPtCalibrated");

    //look up tables
    inMVAweights_edm = iConfig.getParameter<edm::FileInPath> ("inMVA_weights_file");

    // Extract jet pT threshold for Ht, mHt calculation
    jetPtThreshold   = iConfig.getParameter<double> ("JetPtThreshold");
    
    // Determine whether to perform jet pT calibration
    usePtCalib       = iConfig.getParameter<bool> ("UseJetPtCalibration");

    if (usePtCalib == false){
      std::cout << "\nWARNING: Not performing jet pt calibration\n";
    }

}


L1CalibFilterTowerJetProducer::~L1CalibFilterTowerJetProducer()
{
 
}

void
L1CalibFilterTowerJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
   bool evValid = true;

   double ht(0);
   auto_ptr<L1TowerJetCollection> outputCollCen(new L1TowerJetCollection());
   //    auto_ptr< L1TowerJetCollection > outputCollFwd(new L1TowerJetCollection());
   auto_ptr<L1JetParticleCollection> outputExtraCen(new L1JetParticleCollection());
   auto_ptr<L1JetParticleCollection> outputExtraFwd(new L1JetParticleCollection());
   auto_ptr<l1extra::L1EtMissParticleCollection> outputmht(new L1EtMissParticleCollection());
   auto_ptr<double> outRho  (new double());
   auto_ptr<double> outHT   (new double());
   auto_ptr<bool>   useCalib(new bool());

   // Store whether jet pT calibration was applied
   *useCalib = usePtCalib;

   // Read in collection depending on parameters in SLHCCaloTrigger_cfi.py
   edm::Handle<L1TowerJetCollection > PUSubCen;
   iEvent.getByLabel(conf_.getParameter<edm::InputTag>("PUSubtractedCentralJets"), PUSubCen);
   if(!PUSubCen.isValid()){
     edm::LogWarning("MissingProduct") << conf_.getParameter<edm::InputTag>("PUSubtractedCentralJets") << std::endl; 
     evValid = false;
   }


   if( evValid ) {

     ///////////////////////////////////////////////////
     //              JET VALUES 
     ///////////////////////////////////////////////////     
     
     //vector of MHT
     math::PtEtaPhiMLorentzVector mht, upgrade_jet;
   
     //Produce calibrated pt collection: central jets
     for (L1TowerJetCollection::const_iterator il1 = PUSubCen->begin(); il1!= PUSubCen->end() ; ++il1 ){

       L1TowerJet h = (*il1);
              
       double l1Pt_   = il1->Pt();
       double l1wEta_ = il1->WeightedEta();
       double l1wPhi_ = il1->WeightedPhi() ;
       // 'Calibrated' pT, may be uncalibrated if calibration is set to false
       double cal_Pt_;

       if (usePtCalib){ // Get the calibration factors from the MVA lookup table
	 cal_Pt_ = val_pt_cal( l1Pt_ , l1wEta_) * l1Pt_;
       }
       else{            // Perform no calibration
	 cal_Pt_ = l1Pt_;
       }
       
       // Very awkward
       math::PtEtaPhiMLorentzVector p4;
       p4.SetCoordinates(cal_Pt_ , l1wEta_ , l1wPhi_ , il1->p4().M() );
       h.setP4(p4);


       outputCollCen->insert( l1wEta_ , l1wPhi_ , h );
       upgrade_jet.SetCoordinates(cal_Pt_ , l1wEta_ , l1wPhi_ , il1->p4().M() );
       
       // Add to Ht and mHt if jet energy is above threshold
       if( cal_Pt_ > jetPtThreshold ) ht  += cal_Pt_;
       if( cal_Pt_ > jetPtThreshold ) mht += upgrade_jet;
       
       // add jet to L1Extra list
       outputExtraCen->push_back( L1JetParticle( math::PtEtaPhiMLorentzVector( cal_Pt_, l1wEta_, l1wPhi_, 0. ),
						 Ref< L1GctJetCandCollection >(), 0 ));
     }
     

     // create L1Extra object
     math::PtEtaPhiMLorentzVector p4tmp = math::PtEtaPhiMLorentzVector( mht.pt(), 0., mht.phi(), 0. );
	
     L1EtMissParticle l1extraMHT(p4tmp, L1EtMissParticle::kMHT, ht,
				 Ref< L1GctEtMissCollection >(), Ref< L1GctEtTotalCollection >(),
				 Ref< L1GctHtMissCollection >(), Ref< L1GctEtHadCollection >()  ,  0);
     
     outputmht->push_back(l1extraMHT);
     *outHT = ht;
     
     

     iEvent.put(outputCollCen,"CenJets");
     //   iEvent.put(outputCollFwd,"CalibFwdJets");
     iEvent.put(outputExtraCen,"Cen8x8");
     //  iEvent.put(outputExtraFwd,"Fwd8x8");
     iEvent.put(outputmht,"TowerMHT" );
     iEvent.put(outHT,"TowerHT");
     iEvent.put(useCalib,"JetPtCalibrated");
     
     
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1CalibFilterTowerJetProducer::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1CalibFilterTowerJetProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
L1CalibFilterTowerJetProducer::beginRun(edm::Run&, edm::EventSetup const&)
{    
  if (usePtCalib){
    //read in calibration for pt from TMVA
    TMVA_calibration();
  }
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

void L1CalibFilterTowerJetProducer::TMVA_calibration()
{
  std::cout << "Getting lookup from MVA\n";

  reader = new TMVA::Reader("!Color:Silent");
  reader->AddVariable( "l1Pt", &l1Pt);
  reader->AddVariable( "l1Eta", &l1Eta);
//  reader->AddVariable( "l1Phi", &l1Phi);
//  reader->AddVariable( "MVA_Rho", &MVA_Rho );

  std::cout << "Booking MVA reader\n";

  reader->BookMVA("BDT method",inMVAweights_edm.fullPath().c_str());

  std::cout << "MVA reader booked: start processing events.\n";

}

float
L1CalibFilterTowerJetProducer::val_pt_cal(float l1pt, float l1eta)
{
  l1Eta = l1eta;
  l1Pt  = l1pt;

  Float_t val = (reader->EvaluateRegression(TString("BDT method") ))[0];

  //cout<<"l1 pt: " << l1pt <<" corr_pt_1 "<<corr_pt_1<<endl;  
  return val;

}



//define this as a plug-in
DEFINE_FWK_MODULE(L1CalibFilterTowerJetProducer);
