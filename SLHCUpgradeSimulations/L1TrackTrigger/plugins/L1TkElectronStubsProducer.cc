// -*- C++ -*-
//
//
// Producer of a L1TkElectronParticle, for the algorithm matching a L1Track to the L1EG object.
// 	The code here is dummy, just to show how to create an L1TkElectronParticle
//	without a reference to a L1Track  
//      (e.g. when matching the L1EGamma with stubs and not tracks :
//
// The proper producer will be provided by Suchandra & Atanu.
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

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticleFwd.h"

#include "DataFormats/Math/interface/LorentzVector.h"


// for L1Tracks:
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

#include <string>
#include "TMath.h"


using namespace l1extra ;

//
// class declaration
//

class L1TkElectronStubsProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >  L1TkTrackCollectionType;

      explicit L1TkElectronStubsProducer(const edm::ParameterSet&);
      ~L1TkElectronStubsProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      //virtual void beginRun(edm::Run&, edm::EventSetup const&);
      //virtual void endRun(edm::Run&, edm::EventSetup const&);
      //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------
	edm::InputTag L1EGammaInputTag;
	edm::InputTag L1TrackInputTag;
	std::string label;

} ;


//
// constructors and destructor
//
L1TkElectronStubsProducer::L1TkElectronStubsProducer(const edm::ParameterSet& iConfig)
{

   L1EGammaInputTag = iConfig.getParameter<edm::InputTag>("L1EGammaInputTag") ;
   L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
   label = iConfig.getParameter<std::string>("label");
   

   produces<L1TkElectronParticleCollection>(label);
}

L1TkElectronStubsProducer::~L1TkElectronStubsProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkElectronStubsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

 std::auto_ptr<L1TkElectronParticleCollection> result(new L1TkElectronParticleCollection);

 edm::Handle<L1EmParticleCollection> EGammaHandle;
 iEvent.getByLabel(L1EGammaInputTag,EGammaHandle);
 std::vector<L1EmParticle>::const_iterator egIter ;

 edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
 iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);
 L1TkTrackCollectionType::const_iterator trackIter;
 if( !EGammaHandle.isValid() )
        {
          LogError("L1TkElectronStubsProducer")
            << "\nWarning: L1EmParticleCollection with " << L1EGammaInputTag
            << "\nrequested in configuration, but not found in the event. Exit"
            << std::endl;
           return;
        }

 int ieg = 0;
 for (egIter = EGammaHandle->begin();  egIter != EGammaHandle->end(); ++egIter) {

    edm::Ref< L1EmParticleCollection > EGammaRef( EGammaHandle, ieg );
    ieg ++;

    int ibx = egIter -> bx();
    if (ibx != 0) continue;

        float trkisol = -999;       // dummy
        const math::XYZTLorentzVector P4 = egIter -> p4() ;

           edm::Ptr< L1TkTrackType > L1TrackPtrNull;     //  null pointer
           L1TkElectronParticle trkEm( P4,
                                 EGammaRef,
                                 L1TrackPtrNull,  
                                 trkisol );

           	// then one can set the "z" of the electron, as determined by the 
           	// algorithm, via :
	   float z = -999; 	// dummy
           trkEm.setTrkzVtx( z );

	   result -> push_back( trkEm );


 }  // end loop over EGamma objects

 iEvent.put( result, label );

}


// ------------ method called once each job just before starting event loop  ------------
void
L1TkElectronStubsProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkElectronStubsProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkElectronStubsProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkElectronStubsProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkElectronStubsProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkElectronStubsProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkElectronStubsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkElectronStubsProducer);



