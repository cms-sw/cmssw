// -*- C++ -*-
//
//
// dummy producer for a L1TkTauParticle
// The code simply match the L1CaloTaus with the closest L1Track.
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

#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"

#include "DataFormats/Math/interface/LorentzVector.h"


// for L1Tracks:
//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <string>
#include "TMath.h"


using namespace l1extra ;

//
// class declaration
//

class L1TkTauFromCaloProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >  L1TkTrackCollectionType;

      explicit L1TkTauFromCaloProducer(const edm::ParameterSet&);
      ~L1TkTauFromCaloProducer();

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
	
	 //edm::InputTag L1PLInputTag;  // inputTag for PierLuigi's objects

	 edm::InputTag L1TausInputTag;
	 edm::InputTag L1TrackInputTag;	 

        float ZMAX;             // |z_track| < ZMAX in cm
        float CHI2MAX;
        float PTMINTRA;
        float DRmax;

        int nStubsmin ;         // minimum number of stubs 

	bool closest ;


} ;


//
// constructors and destructor
//
L1TkTauFromCaloProducer::L1TkTauFromCaloProducer(const edm::ParameterSet& iConfig)
{

   L1TausInputTag = iConfig.getParameter<edm::InputTag>("L1TausInputTag");
   L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");

   ZMAX = (float)iConfig.getParameter<double>("ZMAX");
   CHI2MAX = (float)iConfig.getParameter<double>("CHI2MAX");
   PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");
   DRmax = (float)iConfig.getParameter<double>("DRmax");
   nStubsmin = iConfig.getParameter<int>("nStubsmin");

   produces<L1TkTauParticleCollection>();
}

L1TkTauFromCaloProducer::~L1TkTauFromCaloProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkTauFromCaloProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;


 std::auto_ptr<L1TkTauParticleCollection> result(new L1TkTauParticleCollection);

 // dummy code. I loop over the L1Taus and pick up the L1Track that is closest
 // in DeltaR.

  edm::Handle< vector<l1extra::L1JetParticle>  > TauHandle;
  iEvent.getByLabel(L1TausInputTag,TauHandle);
  vector<l1extra::L1JetParticle>::const_iterator l1TauIter;

 edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
 iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);
 L1TkTrackCollectionType::const_iterator trackIter;


  int itau = 0;
  for (l1TauIter = TauHandle->begin(); l1TauIter != TauHandle->end(); ++l1TauIter) {

    edm::Ref< L1JetParticleCollection > tauCaloRef( TauHandle, itau );
    itau ++;

        float drmin = 999;

      float eta = l1TauIter -> eta();
      float phi = l1TauIter -> phi();

      int bx = l1TauIter -> bx() ;
      if (bx != 0 ) continue;

	// match the L1Taus with L1Tracks

        int itr = -1;
        int itrack = -1;
        for (trackIter = L1TkTrackHandle->begin(); trackIter != L1TkTrackHandle->end(); ++trackIter) {
	   itr ++ ;
           float Pt = trackIter->getMomentum().perp();
           float z  = trackIter->getPOCA().z();
           if (fabs(z) > ZMAX) continue;
           if (Pt < PTMINTRA) continue;
           float chi2 = trackIter->getChi2();
           if (chi2 > CHI2MAX) continue;

	   std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > >  theStubs = trackIter -> getStubRefs() ;

      	   int tmp_trk_nstub = (int) theStubs.size();
      	   if ( tmp_trk_nstub < nStubsmin) continue;

                float Eta = trackIter->getMomentum().eta();
                float Phi = trackIter->getMomentum().phi();
                float deta = eta - Eta;
                float dphi = phi - Phi;
                if (dphi < 0) dphi = dphi + 2.*TMath::Pi();
                if (dphi > TMath::Pi()) dphi = 2.*TMath::Pi() - dphi;
                float dR = sqrt( deta*deta + dphi*dphi );

			// take the closest track:
                if (dR < drmin) {
                  drmin = dR;
                  itrack = itr;
                }

        }  // end loop over the tracks

        if (drmin < DRmax ) {     // found a L1Track matched to the L1Tau object

            edm::Ptr< L1TkTrackType > L1TrackPtr( L1TkTrackHandle, itrack) ;

            float px = L1TrackPtr -> getMomentum().x();
            float py = L1TrackPtr -> getMomentum().y();
            float pz = L1TrackPtr -> getMomentum().z(); 
            float e = sqrt( px*px + py*py + pz*pz );    // massless particle
            math::XYZTLorentzVector TrackP4(px,py,pz,e);
		// TrackP4 should be the 4-vector of what you define
		// as the tau kinematics. Here I just pick up the PT of
		// the matched track and set m=0. That's certainly no what you would 
		// do e.g. for a 3-prong decay !
            
            float trkisol = -999;       // dummy
            
	    edm::Ptr< L1TkTrackType > L1TrackPtrNull2;     //  null pointer
            edm::Ptr< L1TkTrackType > L1TrackPtrNull3;     //  null pointer

            L1TkTauParticle trkTau( TrackP4,
                                 tauCaloRef,
                                 L1TrackPtr,
				 L1TrackPtrNull2,
				 L1TrackPtrNull3,
                                 trkisol );

	    //trkMu.setDeltaR ( drmin ) ;
            
            result -> push_back( trkTau );

        
         }  // endif drmin < DRmax


   }  // end loop over the L1Taus

 iEvent.put( result );

}

// --------------------------------------------------------------------------------------


// ------------ method called once each job just before starting event loop  ------------
void
L1TkTauFromCaloProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkTauFromCaloProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkTauFromCaloProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkTauFromCaloProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkTauFromCaloProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkTauFromCaloProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkTauFromCaloProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkTauFromCaloProducer);



