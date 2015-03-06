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

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

// for L1Tracks:
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

// Matching Algorithm
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkElectronStubMatchAlgo.h"

#include "DataFormats/Math/interface/deltaPhi.h"

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

      float isolation(const edm::Handle<L1TkTrackCollectionType> & trkHandle, GlobalPoint ep, std::vector<double> & zvals);

      // ----------member data ---------------------------
	edm::InputTag L1EGammaInputTag;
	edm::InputTag L1TrackInputTag;
	edm::InputTag L1StubInputTag;
	edm::InputTag BeamSpotInputTag;
        const edm::ParameterSet conf;
	std::string label;

	float ETmin; 	// min ET in GeV of L1EG objects

	float DRmin;
	float DRmax;
	float PTMINTRA;
	bool PrimaryVtxConstrain;	// use the primary vertex (default = false)
	float DeltaZ;      	// | z_track - z_ref_track | < DeltaZ in cm. 
				// Used only when PrimaryVtxConstrain = True.
	float IsoCut;
	bool RelativeIsolation;
} ;


//
// constructors and destructor
//
L1TkElectronStubsProducer::L1TkElectronStubsProducer(const edm::ParameterSet& iConfig) :
  conf(iConfig)  {

  L1EGammaInputTag = iConfig.getParameter<edm::InputTag>("L1EGammaInputTag") ;
  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
  L1StubInputTag = iConfig.getParameter<edm::InputTag>("L1StubInputTag");
  BeamSpotInputTag = iConfig.getParameter<edm::InputTag>("BeamSpotInputTag");
  label = iConfig.getParameter<std::string>("label");
  
  ETmin = (float)iConfig.getParameter<double>("ETmin");
  
  // parameters for the calculation of the isolation :
  PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");
  DRmin = (float)iConfig.getParameter<double>("DRmin");
  DRmax = (float)iConfig.getParameter<double>("DRmax");
  DeltaZ = (float)iConfig.getParameter<double>("DeltaZ");
  
  // cut applied on the isolation (if this number is <= 0, no cut is applied)
  IsoCut = (float)iConfig.getParameter<double>("IsoCut");
  RelativeIsolation = iConfig.getParameter<bool>("RelativeIsolation");
  
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

    float e_ele   = egIter->energy();
    float eta_ele = egIter->eta();
    float et_ele = 0;
    if (cosh(eta_ele) > 0.0) et_ele = e_ele/cosh(eta_ele);
    else et_ele = -1.0;
    if (ETmin > 0.0 && et_ele <= ETmin) continue;

    std::vector<double> compatibleZpoints;
    unsigned int matchedStubs = L1TkElectronStubMatchAlgo::doMatch(egIter, conf, iSetup, iEvent, compatibleZpoints);
   

    if (matchedStubs > 0) {

      const math::XYZTLorentzVector P4 = egIter -> p4() ;
      GlobalPoint epos = L1TkElectronStubMatchAlgo::calorimeterPosition(egIter->phi(), egIter->eta(), egIter->energy()); 
      float trkisol = isolation(L1TkTrackHandle, epos, compatibleZpoints);
      //      std::cout <<  " Event # " <<  iEvent.id().event() << " EGamma Et " << et_ele  << " Selected Tracklets " << matchedStubs << " Isolation " << trkisol/et_ele << std::endl;
      if (RelativeIsolation && et_ele > 0.0) {   // relative isolation
	trkisol = trkisol  / et_ele;
      }
      edm::Ptr< L1TkTrackType > L1TrackPtrNull; // null pointer

      L1TkElectronParticle trkEm( P4,
				  EGammaRef,
				  L1TrackPtrNull,
				  trkisol );

      if (IsoCut <= 0) {
	// write the L1TkEm particle to the collection, 
	// irrespective of its relative isolation
	result -> push_back( trkEm );
      }	else {
	// the object is written to the collection only
	// if it passes the isolation cut
	if (trkisol <= IsoCut) result -> push_back( trkEm );
      }
     
    }
   
  } // end loop over EGamma objects
  
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
// method to calculate isolation
float 
  L1TkElectronStubsProducer::isolation(const edm::Handle<L1TkTrackCollectionType> & trkHandle, GlobalPoint ep, std::vector<double> & zvals) {
  L1TkTrackCollectionType::const_iterator trackIter;
  
  float er = ep.perp();
  float ez = ep.z();
  float minIso = 999.9;
  
  for (std::vector<double>::iterator iz = zvals.begin(); iz != zvals.end(); iz++) {
    float zcorr = ez - (*iz);
    float theta = 0.0;
    if (zcorr >= 0) theta = atan(er/fabs(zcorr));
    else theta = 3.14 - atan(er/fabs(zcorr));
    float etacorr = -1.0 * TMath::Log(TMath::Tan(theta/2.0));
    
    float sumPt = 0.0;
    for (trackIter = trkHandle->begin(); trackIter != trkHandle->end(); ++trackIter) {
      float dZ = fabs(trackIter->getPOCA().z() - (*iz));       
      float dPhi = reco::deltaPhi(trackIter->getMomentum().phi(), ep.phi());
      float dEta = (trackIter->getMomentum().eta() - etacorr);
      float dR =  sqrt(dPhi*dPhi + dEta*dEta);
      if (dR > DRmin && dR < DRmax && dZ < DeltaZ && trackIter->getMomentum().perp() > PTMINTRA) {
	sumPt += trackIter->getMomentum().perp();
      }
    }
    if (sumPt < minIso) minIso = sumPt;
  }
  return minIso;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkElectronStubsProducer);



