// -*- C++ -*-
//
//
// dummy producer for a L1TkMuonParticle
// This is just an interface, taking the muon objects created
// by PierLuigi's code, and putting them into a collection of
// L1TkMuonParticle.
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

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"

#include "DataFormats/Math/interface/LorentzVector.h"


// for L1Tracks:
//#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <string>
#include "TMath.h"


using namespace l1extra ;

//
// class declaration
//

class L1TkMuonNaiveProducer : public edm::EDProducer {
   public:

   typedef TTTrack< Ref_PixelDigi_ >  L1TkTrackType;
   typedef std::vector< L1TkTrackType >  L1TkTrackCollectionType;

      explicit L1TkMuonNaiveProducer(const edm::ParameterSet&);
      ~L1TkMuonNaiveProducer();

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

	 edm::InputTag L1MuonsInputTag;
	 edm::InputTag L1TrackInputTag;	 

	 float ETAMIN;
	 float ETAMAX;
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
L1TkMuonNaiveProducer::L1TkMuonNaiveProducer(const edm::ParameterSet& iConfig)
{

   //L1PLInputTag = iConfig.getParameter<edm::InputTag>("L1PLInputTag");

   L1MuonsInputTag = iConfig.getParameter<edm::InputTag>("L1MuonsInputTag");
   L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");

   ETAMIN = (float)iConfig.getParameter<double>("ETAMIN");
   ETAMAX = (float)iConfig.getParameter<double>("ETAMAX");
   ZMAX = (float)iConfig.getParameter<double>("ZMAX");
   CHI2MAX = (float)iConfig.getParameter<double>("CHI2MAX");
   PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");
   DRmax = (float)iConfig.getParameter<double>("DRmax");
  nStubsmin = iConfig.getParameter<int>("nStubsmin");
  closest = iConfig.getParameter<bool>("closest");

   produces<L1TkMuonParticleCollection>();
}

L1TkMuonNaiveProducer::~L1TkMuonNaiveProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkMuonNaiveProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;


 std::auto_ptr<L1TkMuonParticleCollection> result(new L1TkMuonParticleCollection);

 // dummy code. I loop over the L1Muons and pick up the L1Track that is closest
 // in DeltaR.

  edm::Handle< vector<l1extra::L1MuonParticle>  > MuonHandle;
  iEvent.getByLabel(L1MuonsInputTag,MuonHandle);
  vector<l1extra::L1MuonParticle>::const_iterator l1MuIter;

 edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
 iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);
 L1TkTrackCollectionType::const_iterator trackIter;


  int imu = 0;
  for (l1MuIter = MuonHandle->begin(); l1MuIter != MuonHandle->end(); ++l1MuIter) {

    edm::Ref< L1MuonParticleCollection > MuRef( MuonHandle, imu );
    imu ++;

        float drmin = 999;
        float ptmax = -1;
	if (ptmax < 0) ptmax = -1;	// dummy

      float eta = l1MuIter -> eta();
      float phi = l1MuIter -> phi();

      float feta = fabs( eta );
      if (feta < ETAMIN) continue;
      if (feta > ETAMAX) continue;

      L1MuGMTExtendedCand cand = l1MuIter -> gmtMuonCand();
      unsigned int quality = cand.quality();
      int bx = l1MuIter -> bx() ;
      if (bx != 0 ) continue;
      if (quality < 3) continue;

	// match the L1Muons with L1Tracks

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

		if (closest) {
			// take the closest track:
                if (dR < drmin) {
                  drmin = dR;
                  itrack = itr;
                }
		}
		else {
			// or take the leading track within a cone 
		if (dR < DRmax) {
		  if (Pt > ptmax) {
			ptmax = Pt;
			itrack = itr;
			drmin = dR;
		  }
		}
		}

        }  // end loop over the tracks

        if (drmin < DRmax ) {     // found a L1Track matched to the L1Muon object

            edm::Ptr< L1TkTrackType > L1TrackPtr( L1TkTrackHandle, itrack) ;

            float px = L1TrackPtr -> getMomentum().x();
            float py = L1TrackPtr -> getMomentum().y();
            float pz = L1TrackPtr -> getMomentum().z(); 
            float e = sqrt( px*px + py*py + pz*pz );    // massless particle
            math::XYZTLorentzVector TrackP4(px,py,pz,e);
            
            float trkisol = -999;       // dummy
            
            L1TkMuonParticle trkMu( TrackP4,
                                 MuRef,
                                 L1TrackPtr,
                                 trkisol );

	    //trkMu.setDeltaR ( drmin ) ;
            
            result -> push_back( trkMu );

        
         }  // endif drmin < DRmax


   }  // end loop over the L1Muons

	// PL: the muon objects from PierLuigi
/*
 edm::Handle<XXXCollection> XXXHandle;
 iEvent.getByLabel(L1PLInputTag,XXXHandle);
 std::vector<XXX>::const_iterator muIter ;

 if (!XXXHandle.isValid() ) {
          LogError("L1TkMuonNaiveProducer")
            << "\nWarning: L1XXXCollectionType with " << L1PLInputTag
            << "\nrequested in configuration, but not found in the event. Exit."
            << std::endl;
           return;
 }

	// Now loop over the muons of Pierluigi 

 int imu = 0;
 for (muIter = XXXHandle->begin();  muIter != XXXHandle->end(); ++muIter) {

    edm::Ref< XXXCollection > muRef( XXXHandle, imu );
    imu ++;

    // int bx = egIter -> bx() ;	// if PL's objects have a bx method
    int bx = 0;    // else...

    if (bx == 0) {

	edm::Ptr< L1TkTrackType > L1TrackPtr ;

	// PL : get the matched L1Track from PL's object
	// L1TrackPtr  = muRef -> getRefToTheL1Track() ;
	
            float px = L1TrackPtr -> getMomentum().x();
            float py = L1TrackPtr -> getMomentum().y();
            float pz = L1TrackPtr -> getMomentum().z();
            float e = sqrt( px*px + py*py + pz*pz );    // massless particle
            math::XYZTLorentzVector TrackP4(px,py,pz,e);

	// the code may calculate a tracker-based isolation variable,
	// or pick it up from PL's object if it is there.
	// for the while, dummy.
	float trkisol = -999;


	L1TkMuonParticle trkMu(  P4,
				// muRef,  
				L1TrackPtr,	
				trkisol );
    
     }  // endif bx==0

 }  // end loop over Pierluigi's objects
*/

 iEvent.put( result );

}

// --------------------------------------------------------------------------------------


// ------------ method called once each job just before starting event loop  ------------
void
L1TkMuonNaiveProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkMuonNaiveProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkMuonNaiveProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkMuonNaiveProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkMuonNaiveProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkMuonNaiveProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkMuonNaiveProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkMuonNaiveProducer);



