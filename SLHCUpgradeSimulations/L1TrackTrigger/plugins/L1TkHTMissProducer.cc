// -*- C++ -*-
//
//
// Original Author:  Emmanuelle Perez,40 1-A28,+41227671915,
//         Created:  Tue Nov 12 17:03:19 CET 2013
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticleFwd.h"


using namespace l1extra;

//
// class declaration
//

class L1TkHTMissProducer : public edm::EDProducer {
   public:

      explicit L1TkHTMissProducer(const edm::ParameterSet&);
      ~L1TkHTMissProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      //virtual void endRun(edm::Run&, edm::EventSetup const&);
      //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------

        edm::InputTag L1TkJetInputTag;

	bool PrimaryVtxConstrain;
        edm::InputTag L1VertexInputTag; // used only when PrimaryVtxConstrain = True.

	float DeltaZ;


};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
L1TkHTMissProducer::L1TkHTMissProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   //now do what ever other initialization is needed
  
  L1TkJetInputTag = iConfig.getParameter<edm::InputTag>("L1TkJetInputTag");
  DeltaZ = (float)iConfig.getParameter<double>("DeltaZ");

  L1VertexInputTag = iConfig.getParameter<edm::InputTag>("L1VertexInputTag") ;
  PrimaryVtxConstrain = iConfig.getParameter<bool>("PrimaryVtxConstrain");

  produces<L1TkHTMissParticleCollection>();

}


L1TkHTMissProducer::~L1TkHTMissProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1TkHTMissProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
 
 std::auto_ptr<L1TkHTMissParticleCollection> result(new L1TkHTMissParticleCollection);

 edm::Handle<L1TkPrimaryVertexCollection> L1VertexHandle;
 iEvent.getByLabel(L1VertexInputTag,L1VertexHandle);
 std::vector<L1TkPrimaryVertex>::const_iterator vtxIter;

 edm::Handle<L1TkJetParticleCollection> L1TkJetsHandle;
 iEvent.getByLabel(L1TkJetInputTag,L1TkJetsHandle);
 std::vector<L1TkJetParticle>::const_iterator jetIter;


 if ( ! L1TkJetsHandle.isValid() ) {
          LogError("L1TkHTMissProducer")
            << "\nWarning: L1TkJetParticleCollection with " << L1TkJetInputTag
            << "\nrequested in configuration, but not found in the event. Exit"
            << std::endl;
           return;
 }

 float zvtx = -999;

 edm::Ref< L1TkPrimaryVertexCollection > L1VtxRef; 	// null reference

 if ( PrimaryVtxConstrain ) {
     if( !L1VertexHandle.isValid() )
            {
              LogError("L1TkHTMissProducer")
                << "\nWarning: L1TkPrimaryVertexCollection with " << L1VertexInputTag
                << "\nrequested in configuration, but not found in the event. Exit."
                << std::endl;
	       return ;
            }
      else {
                std::vector<L1TkPrimaryVertex>::const_iterator vtxIter = L1VertexHandle->begin();
                   // by convention, the first vertex in the collection is the one that should
                   // be used by default
                zvtx = vtxIter -> getZvertex();
	        int ivtx = 0;
	        edm::Ref< L1TkPrimaryVertexCollection > vtxRef( L1VertexHandle, ivtx );
	        L1VtxRef = vtxRef;
      }
 }  //endif PrimaryVtxConstrain


    float sumPx = 0;
    float sumPy = 0;
    float etTot = 0;	// HT

 for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {

    int ibx = jetIter -> bx();
    if (ibx != 0) continue;

    float px = jetIter -> px();
    float py = jetIter -> py();
    float et = jetIter -> et();
    float jetVtx = jetIter -> getJetVtx();
    
	// vertex consistency requirement :
	// here I use the zvtx from the primary vertex
    bool VtxRequirement = fabs( jetVtx - zvtx ) < DeltaZ ;

    if (VtxRequirement) {
	sumPx += px;
	sumPy += py;
	etTot += et;
    }
 }  // end loop over jets

     float et = sqrt( sumPx*sumPx + sumPy*sumPy );	// HTM
     math::XYZTLorentzVector missingEt( -sumPx, -sumPy, 0, et); 

     edm::RefProd<L1TkJetParticleCollection> jetCollRef(L1TkJetsHandle) ;

     L1TkHTMissParticle tkHTM( missingEt,
				etTot,
                                jetCollRef,
                                L1VtxRef );

/*
	// in case no primary vtx is used, need to set the zvtx that was
	// used in the consistency requirement (e.g. the zvtx of the
	// leading jet) 

     if (! PrimaryVtxConstrain) {
	tkHTM -> setVtx ( zvtx_used );
     }
*/

  result -> push_back(  tkHTM );

  iEvent.put( result);
}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TkHTMissProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TkHTMissProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
L1TkHTMissProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{


}
 
// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkHTMissProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkHTMissProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkHTMissProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkHTMissProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkHTMissProducer);
