// -*- C++ -*-
//
//
// Producer for a L1TkTauParticle from Track seeds.
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

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"




// for TTTracks:
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkTauEtComparator.h"


#include <string>
#include "TMath.h"


using namespace l1extra ;

//
// class declaration
//

class L1TkTauFromL1TrackProducer : public edm::EDProducer {

public:

  typedef  TTTrack< Ref_PixelDigi_ >          L1TkTrackType;
  typedef std::vector< L1TkTrackType >         L1TkTrackCollectionType;

  explicit L1TkTauFromL1TrackProducer(const edm::ParameterSet&);
  ~L1TkTauFromL1TrackProducer();

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
L1TkTauFromL1TrackProducer::L1TkTauFromL1TrackProducer(const edm::ParameterSet& iConfig)
{

  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");

  ZMAX = (float)iConfig.getParameter<double>("ZMAX");
  CHI2MAX = (float)iConfig.getParameter<double>("CHI2MAX");
  PTMINTRA = (float)iConfig.getParameter<double>("PTMINTRA");
  DRmax = (float)iConfig.getParameter<double>("DRmax");
  nStubsmin = iConfig.getParameter<int>("nStubsmin");

  produces<L1TkTauParticleCollection>();

}

L1TkTauFromL1TrackProducer::~L1TkTauFromL1TrackProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkTauFromL1TrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;



  std::auto_ptr<L1TkTauParticleCollection> result(new L1TkTauParticleCollection);


  edm::Handle<L1TkTrackCollectionType> L1TkTrackHandle;
  iEvent.getByLabel(L1TrackInputTag, L1TkTrackHandle);
  L1TkTrackCollectionType::const_iterator trackIter;

  edm::Ptr< L1TkTrackType > l1tracks[1000];     

  math::XYZTLorentzVector p4[1000];
  double z[1000];
  double chi2[1000]; 
  int nstub[1000];

  int ntrack=0;

  int itrack=-1;
  for (trackIter = L1TkTrackHandle->begin(); trackIter != L1TkTrackHandle->end(); ++trackIter) {
    itrack++;

    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > theStubs = trackIter ->getStubRefs();

    z[ntrack]=trackIter->getPOCA().z();
    double px=trackIter->getMomentum().x();
    double py=trackIter->getMomentum().y();
    double pz=trackIter->getMomentum().z();
    double e=sqrt(px*px+py*py+pz*pz+0.14*0.14);


    nstub[ntrack]=theStubs.size();
    chi2[ntrack]=trackIter->getChi2();
    if (chi2[ntrack]>100) continue;
    

    edm::Ptr< L1TkTrackType > L1TrackPtr( L1TkTrackHandle, itrack) ;
    l1tracks[ntrack]=L1TrackPtr;
    math::XYZTLorentzVector p4tmp(px,py,pz,e);
    p4[ntrack++]=p4tmp;

  }


  //Let's look here for 1 prong taus

  for (int i=0;i<ntrack;i++) {
    if (nstub[i]<5) continue;
    if (chi2[i]>10) continue;
    if (p4[i].Pt()<10.0) continue;

    bool foundHigher=false;
    for (int j=0;j<ntrack;j++) {
      if (j==i) continue;
      if (p4[j].Pt()>p4[i].Pt()){
	double dphi=p4[i].Phi()-p4[j].Phi();
	if (dphi>3.141592) dphi-=2*3.141592;
	if (dphi<-3.141592) dphi+=2*3.141592;
	double deta=p4[i].Eta()-p4[j].Eta();
	double dR=sqrt(dphi*dphi+deta*deta);
	if (dR<0.3) foundHigher=true;
      } 
    }

    if (foundHigher) continue;


    math::XYZTLorentzVector p4tot=p4[i];

    double ptsum=0.0;
    for (int l=0;l<ntrack;l++){
      if (l==i) continue;
      if (fabs(z[i]-z[l]) > 0.8 ) continue;
      double dphi=p4[i].Phi()-p4[l].Phi();
      if (dphi>3.141592) dphi-=2*3.141592;
      if (dphi<-3.141592) dphi+=2*3.141592;
      double deta=p4[i].Eta()-p4[l].Eta();
      double dR=sqrt(dphi*dphi+deta*deta);
      if (dR>0.3) continue;
      ptsum+=p4[l].Pt();
    }

    double reliso=ptsum/p4tot.Pt();
 

    if (reliso>0.999) reliso=0.999; 
    if (reliso<0.25) {
      edm::Ptr< L1TkTrackType > L1TrackPtrNull2;     //  null pointer
      edm::Ptr< L1TkTrackType > L1TrackPtrNull3;     //  null pointer
      edm::Ref< L1JetParticleCollection > tauCaloRef; // null pointer

      L1TkTauParticle trkTau( p4[i],
			      tauCaloRef,
			      l1tracks[i],
			      L1TrackPtrNull2,
			      L1TrackPtrNull3,
			      reliso );


      result -> push_back( trkTau );

    }     

  }

 

  //Let's look here for 3 prong taus

  for (int i=0;i<ntrack;i++) {
    if (p4[i].Pt()<10.0) continue;
    bool foundHigher=false;
    for (int j=0;j<ntrack;j++) {
      if (j==i) continue;
      if (p4[j].Pt()>p4[i].Pt()){
	double dphi=p4[i].Phi()-p4[j].Phi();
	if (dphi>3.141592) dphi-=2*3.141592;
	if (dphi<-3.141592) dphi+=2*3.141592;
	double deta=p4[i].Eta()-p4[j].Eta();
	double dR=sqrt(dphi*dphi+deta*deta);
	if (dR<0.3) foundHigher=true;
      } 
    }
    if (foundHigher) continue;
    
    
    double minRelIso=1e10;

    int jmin=-1;
    int kmin=-1;

    for (int j=0;j<ntrack;j++) {
      if (j==i) continue;
      if (p4[j].Pt()>p4[i].Pt()) continue;
      math::XYZTLorentzVector p4sum=p4[i]+p4[j];
      if ( p4sum.M() > 1.777 )  continue;
      for (int k=0;k<j;k++) {
	if (k==i) continue;
	if (p4[k].Pt()>p4[i].Pt()) continue;
	math::XYZTLorentzVector p4tot=p4sum+p4[k];	 
	if ( p4tot.M() > 1.777 )  continue;
	if ( fabs(z[i]-z[j]) > 1.0 ) continue;
	if ( fabs(z[i]-z[k]) > 1.0 ) continue;
	double ptsum=0.0;
	for (int l=0;l<ntrack;l++){
	  if (l==i) continue;
	  if (l==j) continue;
	  if (l==k) continue;
	  if (fabs(z[i]-z[l]) > 0.8 ) continue;
	  double dphi=p4[i].Phi()-p4[l].Phi();
	  if (dphi>3.141592) dphi-=2*3.141592;
	  if (dphi<-3.141592) dphi+=2*3.141592;
	  double deta=p4[i].Eta()-p4[l].Eta();
	  double dR=sqrt(dphi*dphi+deta*deta);
	  if (dR>0.3) continue;
	  ptsum+=p4[l].Pt();
	}
	double reliso=ptsum/p4tot.Pt();
	if (reliso<minRelIso) {
	  minRelIso=reliso;
	  jmin=j;
	  kmin=k;
	}
      }
    }
    
     
    if (minRelIso>1e9) {
      //Did not find  tracks...
      continue;
    }
      

    if (minRelIso>0.999) minRelIso=0.999; 
    if (minRelIso<0.25) {
      edm::Ref< L1JetParticleCollection > tauCaloRef; // null pointer
      
      L1TkTauParticle trkTau( p4[i]+p4[jmin]+p4[kmin],
			      tauCaloRef,
			      l1tracks[i],
			      l1tracks[jmin],
			      l1tracks[kmin],
			      minRelIso );
      
      result -> push_back( trkTau );



    }     

  }

   sort( result->begin(), result->end(), L1TkTau::PtComparator() );
           
 iEvent.put( result );

}

// --------------------------------------------------------------------------------------


// ------------ method called once each job just before starting event loop  ------------
void
L1TkTauFromL1TrackProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkTauFromL1TrackProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkTauFromL1TrackProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkTauFromL1TrackProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkTauFromL1TrackProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkTauFromL1TrackProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkTauFromL1TrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkTauFromL1TrackProducer);



