//This producer creates a PU subtracted collection of upgrade jets in both slhc::L1TowerJet format
//and l1extra::L1JetParticle format
//
//It takes rho produced by L1TowerJetPUSubtraction producer and applies PU subtraction to input collection which
//is filtered, uncalibrated jet collection
//
//Calibration should be done after this step is completed

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
};

L1TowerJetPUSubtractedProducer::L1TowerJetPUSubtractedProducer(const edm::ParameterSet& iConfig):
conf_(iConfig)
{
    produces<L1TowerJetCollection>("PUSubCenJets");
    produces< L1JetParticleCollection >( "PUSubCen8x8" ) ;
}


L1TowerJetPUSubtractedProducer::~L1TowerJetPUSubtractedProducer()
{
}




// ------------ method called to produce the data  ------------
void
L1TowerJetPUSubtractedProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    
    bool evValid =true;
    auto_ptr< L1TowerJetCollection > outputCollCen(new L1TowerJetCollection());
    produces<L1TowerJetCollection>("CalibFwdJets");
    auto_ptr< L1JetParticleCollection > outputExtraCen(new L1JetParticleCollection());


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

    if( evValid ) {

      //get rho from the producer L1TowerJetPUSubtraction
      //This value is calibrated to offline calo rho
      double cal_rhoL1 = *calRho;

      ///////////////////////////////////////////////////
      //              JET VALUES 
      ///////////////////////////////////////////////////     
      
      math::PtEtaPhiMLorentzVector upgrade_jet;
    
      //Produce calibrated pt collection: central jets
      for (L1TowerJetCollection::const_iterator il1 = UnCalibCen->begin();
           il1!= UnCalibCen->end() ;
           ++il1 ){

          L1TowerJet h=(*il1);

    //      float l1Eta_ = il1->p4().eta();
    //      float l1Phi_ = il1->p4().phi();
          float l1Pt_  = il1->p4().Pt();

          //weighted eta is still not correct
          //change the contents out p4, upgrade_jet when it is corrected
          float l1wEta_ = il1->WeightedEta();
          float l1wPhi_ = il1->WeightedPhi() ;


          //This is just for 8x8 circular jets: change if using different jets
          double areaPerJet = 52 * (0.087 * 0.087) ;
          //PU subtraction
          float l1Pt_PUsub_ = l1Pt_ - (cal_rhoL1 * areaPerJet);
          
          //only keep jet if pt > 0 after PU sub 
          if(l1Pt_PUsub_>0.1){

            math::PtEtaPhiMLorentzVector p4;

            //use weighted eta and phi: these are energy weighted 
            p4.SetCoordinates(l1Pt_PUsub_ , l1wEta_ , l1wPhi_ , il1->p4().M() );

            h.setP4(p4);
            outputCollCen->insert( l1wEta_ , l1wPhi_ , h );
            upgrade_jet.SetCoordinates( l1Pt_PUsub_ , l1wEta_ , l1wPhi_ , il1->p4().M() );

            // add jet to L1Extra list
            outputExtraCen->push_back( L1JetParticle( math::PtEtaPhiMLorentzVector( 
                                                        l1Pt_PUsub_,
									            	    l1wEta_,
									            	    l1wPhi_,
									            	    0. ),
						                Ref< L1GctJetCandCollection >(), 0 )
				                       );
          }
        }
      }

    //this is the slhc collection containing extra information
    iEvent.put(outputCollCen,"PUSubCenJets");
    //this is the l1extra collection containing the same jet vector as in slhc collection
    iEvent.put(outputExtraCen,"PUSubCen8x8");

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
