// -*- C++ -*-
//
//
// Producer for a L1TkTauParticle from L1 Tracks and EG Crystal info.
//
// 

// system include files
#include <memory>

// User include files
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

#include "SimDataFormats/SLHC/interface/L1EGCrystalCluster.h"


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

class L1TkEmTauProducer : public edm::EDProducer {

public:

  typedef  TTTrack< Ref_PixelDigi_ >          L1TkTrackType;
  typedef std::vector< L1TkTrackType >         L1TkTrackCollectionType;

  explicit L1TkEmTauProducer(const edm::ParameterSet&);
  ~L1TkEmTauProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  double dR(math::XYZTLorentzVector p41,math::XYZTLorentzVector p42);

  //virtual void beginRun(edm::Run&, edm::EventSetup const&);
  //virtual void endRun(edm::Run&, edm::EventSetup const&);
  //virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
  //virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

  // ----------member data ---------------------------
	
  edm::InputTag L1TrackInputTag;	 
  edm::InputTag L1EmInputTag;	 

  double ptleadcut=10.0;   //Minimum pt of lead track
  double ptleadcone=0.3;  //Cone in which there is no higher pt track
  double masscut=1.77;  //Mass cut
  double emptcut=7.0;   //minimum emcut
  double trketacut=2.3; //largest lead trk eta
  double pttotcut=25.0; //Total transverse energy cut
  double isocone=0.5;   //size of isolation cone
  double isodz=1.0;    //use tracks within one cm of lead track
  double relisocut=0.15; //cut on reliso;
  double chisqcut=40.0;
  int nstubcut=5; //minimum number of stubs
  double dzcut=0.8; //Look for tau tracks near lead track


} ;


//
// constructors and destructor
//
L1TkEmTauProducer::L1TkEmTauProducer(const edm::ParameterSet& iConfig)
{

  L1TrackInputTag = iConfig.getParameter<edm::InputTag>("L1TrackInputTag");
  L1EmInputTag = iConfig.getParameter<edm::InputTag>("L1EmInputTag");

  ptleadcut=iConfig.getParameter<double>("ptleadcut");
  ptleadcone=iConfig.getParameter<double>("ptleadcone");
  masscut=iConfig.getParameter<double>("masscut");
  emptcut=iConfig.getParameter<double>("emptcut");
  trketacut=iConfig.getParameter<double>("trketacut");
  pttotcut=iConfig.getParameter<double>("pttotcut");
  isocone=iConfig.getParameter<double>("isocone");
  isodz=iConfig.getParameter<double>("isodz");
  relisocut=iConfig.getParameter<double>("relisocut");
  chisqcut=iConfig.getParameter<double>("chisqcut");
  nstubcut=iConfig.getParameter<int>("nstubcut");
  dzcut=iConfig.getParameter<double>("dzcut");

  produces<L1TkTauParticleCollection>();

}

L1TkEmTauProducer::~L1TkEmTauProducer() {
}

// ------------ method called to produce the data  ------------
void
L1TkEmTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
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

  ///sort tracks based on pt - use simple bubble sort...

  bool more=false;

  do {
      
    more=false;

    for (int i=0;i<ntrack-1;i++) {
      if (p4[i].Pt()<p4[i+1].Pt()){
	more=true;
	edm::Ptr< L1TkTrackType > tmpl1=l1tracks[i];
	l1tracks[i]=l1tracks[i+1];
	l1tracks[i+1]=tmpl1;
	
	math::XYZTLorentzVector tmpp4=p4[i];
	p4[i]=p4[i+1];
	p4[i+1]=tmpp4;

	double tmpz=z[i];
	z[i]=z[i+1];
	z[i+1]=tmpz;

	double tmpchi2=chi2[i];
	chi2[i]=chi2[i+1];
	chi2[i+1]=tmpchi2;

	int tmpnstub=nstub[i];
	nstub[i]=nstub[i+1];
	nstub[i+1]=tmpnstub;

      }
    }
  } while (more);


  math::XYZTLorentzVector p4em[1000];
  int nem=0;

  edm::Handle<l1slhc::L1EGCrystalClusterCollection> L1EmHandle;
  iEvent.getByLabel(L1EmInputTag, L1EmHandle);
  std::vector<l1slhc::L1EGCrystalCluster>::const_iterator egIter;

  if ( L1EmHandle.isValid() ) {
    //std::cout << "Found L1EmParticles"<<std::endl;
    for (egIter=L1EmHandle->begin();egIter!=L1EmHandle->end();++egIter) {
      double theta=2*atan(exp(-egIter->eta()));
      double px=egIter->pt()*cos(egIter->phi());
      double py=egIter->pt()*sin(egIter->phi());
      double pz=egIter->pt()/tan(theta);
      double e=sqrt(px*px+py*py+pz*pz);
      math::XYZTLorentzVector p4tmp(px,py,pz,e);
      p4em[nem++]=p4tmp;
    }
  }
  else {
    std::cout << "Did not find L1EmParticles"<<std::endl;
  }


 
  for (int it=0; it<ntrack; it++) {

    //check if good lead track
    if (chi2[it]>chisqcut) continue;
    if (nstub[it]<nstubcut) continue;
    if (p4[it].Pt()<ptleadcut) continue;
    if (fabs(p4[it].Eta())>trketacut) continue;


    //Found candidate lead track
    //Now check if there ishigher pTtrack nearby

    bool higherpt=false;
    for (int it2=0; it2<ntrack; it2++) {
      if (it2==it) continue;
      if (nstub[it2]<nstubcut) continue;
      if (p4[it2].Pt()<p4[it].Pt()) continue;
      if (dR(p4[it2],p4[it])<ptleadcone) {
	higherpt=true;
      }
    }
    if (higherpt) continue;

    //Now we have a lead track
    math::XYZTLorentzVector p4lead=p4[it];


    //Now add other tracks
    math::XYZTLorentzVector p4tau=p4lead;
    vector<int> tracks;
    tracks.push_back(it);

    for (int it2=0; it2<ntrack; it2++) {
      if (it2==it) continue;
      if (fabs(z[it]-z[it2])>dzcut) continue;
      //if (dR(p4lead,p4[it2])>ptleadcone/2.0) continue;
      math::XYZTLorentzVector p4trk=p4[it2];
      if ((p4trk+p4tau).M()<masscut) {
	tracks.push_back(it2); 
	p4tau+=p4trk;
      }
    }


    //Found lead track + other tau tracks
    //Next calculate sumpt of tracks in iso cone.

    double isoptsum=0.0;

    for (int it2=0; it2<ntrack; it2++) {
      bool used=false;
      for (int j=0;j<(int)tracks.size();j++){
	if (tracks[j]==it2) used=true;
      } 
      if (used) continue;
      if (dR(p4lead,p4[it2])<isocone) {
	if (fabs(z[it]-z[it2])<isodz) {
	  isoptsum+=p4[it2].Pt();
	}
      }
      
      
    }

    //Next look for photon(s)

    double ptem=0.0;
    for (int it2=0; it2<nem; it2++) {
      if (p4em[it2].Pt()<emptcut) continue;
      math::XYZTLorentzVector p4gamma=p4em[it2];
      if ((p4tau+p4gamma).M()<masscut) {
	p4tau+=p4gamma;
	ptem+=p4gamma.Pt();
      }
      
    }  
    
    if (p4tau.Pt()<pttotcut) {
      continue;
    }
        
    double reliso=isoptsum/p4tau.Pt();

    if (reliso>relisocut) continue;


    edm::Ref< L1JetParticleCollection > tauCaloRef; // null pointer

    edm::Ptr< L1TkTrackType > L1Track2;     //  null pointer
    edm::Ptr< L1TkTrackType > L1Track3;     //  null pointer

    if (tracks.size()>1) L1Track2=l1tracks[tracks[1]];
    if (tracks.size()>2) L1Track3=l1tracks[tracks[2]];


    //std::cout << "Creating tau"<<std::endl;
    L1TkTauParticle trkTau( p4tau,
			    tauCaloRef,
			    l1tracks[it],
			    L1Track2,
			    L1Track3,
			    reliso );
    


      
    result -> push_back( trkTau );

  }
  
  sort( result ->begin(), result ->end(), L1TkTau::EtComparator() );

  iEvent.put( result );

}

double L1TkEmTauProducer::dR(math::XYZTLorentzVector p41,math::XYZTLorentzVector p42) {

  double dPhi=p41.Phi()-p42.Phi();
  static const double pi=4*atan(1.0);
  
  if (dPhi>pi) dPhi-=2*pi;
  if (dPhi<-pi) dPhi+=2*pi;
  
  double deta=p41.Eta()-p42.Eta();

  double dr=sqrt(dPhi*dPhi+deta*deta);
  
  
  return dr;
  
}



// --------------------------------------------------------------------------------------


// ------------ method called once each job just before starting event loop  ------------
void
L1TkEmTauProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1TkEmTauProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
L1TkEmTauProducer::beginRun(edm::Run& iRun, edm::EventSetup const& iSetup)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
L1TkEmTauProducer::endRun(edm::Run&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
L1TkEmTauProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
L1TkEmTauProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TkEmTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TkEmTauProducer);



