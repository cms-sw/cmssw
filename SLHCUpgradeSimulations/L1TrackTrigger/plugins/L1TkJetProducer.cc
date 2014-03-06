// -*- C++ -*-
//
//
// Producer of a L1TkJetParticle.
// Dummy code below. To be filled by Louise.
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

#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleFwd.h"

#include "DataFormats/Math/interface/LorentzVector.h"


// for L1Tracks:
//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <string>
#include "TMath.h"


using namespace l1extra ;

//
// class declaration
//

class L1TkJetProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >    L1TkTrackCollectionType;

      explicit L1TkJetProducer(const edm::ParameterSet&);
      ~L1TkJetProducer();

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
	edm::InputTag L1CentralJetInputTag;
	edm::InputTag L1TrackInputTag;

} ;


//
// constructors and destructor
//
L1TkJetProducer::L1TkJetProducer(const edm::ParameterSet& iConfig)
{

   L1CentralJetInputTag = iConfig.getParameter<edm::InputTag>("L1CentralJetInputTag") ;
   L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
   
   produces<L1TkJetParticleCollection>("Central");

}

L1TkJetProducer::~L1TkJetProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

 std::auto_ptr<L1TkJetParticleCollection> cenTkJets(new L1TkJetParticleCollection);

 edm::Handle<L1JetParticleCollection> CentralJetHandle;
 iEvent.getByLabel(L1CentralJetInputTag,CentralJetHandle);
 std::vector<L1JetParticle>::const_iterator jetIter ;

 edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
 iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);
 L1TkTrackCollectionType::const_iterator trackIter;

	// central jets (i.e. |eta| < 3)

 if( !CentralJetHandle.isValid() )
	{
	  LogError("L1TkJetProducer")
	    << "\nWarning: L1JetParticleCollection with " << L1CentralJetInputTag
	    << "\nrequested in configuration, but not found in the event."
	    << std::endl;
	}
 else {

    int ijet = 0;
    for (jetIter = CentralJetHandle->begin();  jetIter != CentralJetHandle->end(); ++jetIter) {

       edm::Ref< L1JetParticleCollection > JetRef( CentralJetHandle, ijet );
       ijet ++;

       int ibx = jetIter -> bx();
       if (ibx != 0) continue;

	   // calculate the vertex of the jet. Here dummy.
	   float jetvtx = -999;

           const math::XYZTLorentzVector P4 = jetIter -> p4() ;
           L1TkJetParticle trkJet(  P4,
                                   JetRef,
				   jetvtx );

	       cenTkJets -> push_back( trkJet );

    }  // end loop over Jet objects
 } // endif CentralJetHandle.isValid()



 iEvent.put( cenTkJets, "Central" );

}


// ------------ method called once each job just before starting event loop  ------------
void
L1TkJetProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkJetProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkJetProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkJetProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkJetProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkJetProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkJetProducer);



