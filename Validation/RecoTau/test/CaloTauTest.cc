#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorAgainstElectron.h"

#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <memory>
#include <string>
#include <iostream>

#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TH1.h>

using namespace edm;
using namespace reco; 
using namespace std;

class CaloTauTest : public EDAnalyzer {
public:
  explicit CaloTauTest(const ParameterSet&);
  ~CaloTauTest() {}
  virtual void analyze(const Event& iEvent,const EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();
private:
  string CaloTauProducer_;
  string CaloTauDiscriminatorByIsolationProducer_;
  string CaloTauDiscriminatorAgainstElectronProducer_;
  int nEvent;
};

CaloTauTest::CaloTauTest(const ParameterSet& iConfig){
  CaloTauProducer_                            = iConfig.getParameter<string>("CaloTauProducer");
  CaloTauDiscriminatorByIsolationProducer_    = iConfig.getParameter<string>("CaloTauDiscriminatorByIsolationProducer");
  CaloTauDiscriminatorAgainstElectronProducer_    = iConfig.getParameter<string>("CaloTauDiscriminatorAgainstElectronProducer");
  nEvent=0;
}

void CaloTauTest::beginJob(){}

void CaloTauTest::analyze(const Event& iEvent, const EventSetup& iSetup){
  cout<<endl;
  cout<<"********"<<endl;
  cout<<"Event number "<<nEvent++<<endl;
  
  Handle<CaloTauCollection> theCaloTauHandle;
  iEvent.getByLabel(CaloTauProducer_,theCaloTauHandle);
  
  Handle<CaloTauDiscriminatorByIsolation> theCaloTauDiscriminatorByIsolation;
  iEvent.getByLabel(CaloTauDiscriminatorByIsolationProducer_,theCaloTauDiscriminatorByIsolation);

  Handle<CaloTauDiscriminatorAgainstElectron> theCaloTauDiscriminatorAgainstElectron;
  iEvent.getByLabel(CaloTauDiscriminatorAgainstElectronProducer_,theCaloTauDiscriminatorAgainstElectron);

  cout<<"***"<<endl;
  cout<<"Found "<<theCaloTauHandle->size()<<" had. tau-jet candidates"<<endl;
  int i_CaloTau=0;
  for (CaloTauCollection::size_type iCaloTau=0;iCaloTau<theCaloTauHandle->size();iCaloTau++) {
    CaloTauRef theCaloTau(theCaloTauHandle,iCaloTau);
    //Prints out some quantities
    cout<<"***"<<endl;
    cout<<"Jet Number "<<i_CaloTau<<endl;
    cout<<"CaloDiscriminatorByIsolation value "<<(*theCaloTauDiscriminatorByIsolation)[theCaloTau]<<endl;
    cout<<"CaloDiscriminatorAgainstElectron value "<<(*theCaloTauDiscriminatorAgainstElectron)[theCaloTau]<<endl;
    cout<<"Pt of the CaloTau (GeV/c) "<<(*theCaloTau).pt()<<endl;
    cout<<"InvariantMass of the Tracks + neutral ECAL Island BasicClusters system (GeV/c2) "<<(*theCaloTau).alternatLorentzVect().M()<<endl;
    cout<<"InvariantMass of the Tracks system (GeV/c2) "<<(*theCaloTau).TracksInvariantMass()<<endl;
    cout<<"Charge of the CaloTau "<<(*theCaloTau).charge()<<endl;
    cout<<"Inner point position (x,y,z) of the CaloTau ("<<(*theCaloTau).vx()<<","<<(*theCaloTau).vy()<<","<<(*theCaloTau).vz()<<")"<<endl;
    cout<<"Et of the highest Et HCAL hit (GeV) "<<(*theCaloTau).maximumHCALhitEt()<<endl;
    cout<<"# Tracks "<<(*theCaloTau).caloTauTagInfoRef()->Tracks().size()<<endl;
    cout<<"# neutral ECAL Island algo. BasicClusters "<<(*theCaloTau).caloTauTagInfoRef()->neutralECALBasicClusters().size()<<endl;
    TrackRef theLeadTk=(*theCaloTau).leadTrack();
    if(!theLeadTk){
      cout<<"No Lead Tk "<<endl;
    }else{
      cout<<"Lead Tk pt (GeV/c) "<<(*theLeadTk).pt()<<endl;
      cout<<"Lead Tk signed transverse impact parameter significance "<<(*theCaloTau).leadTracksignedSipt()<<endl;
      cout<<"InvariantMass of the signal Tracks system (GeV/c2) "<<(*theCaloTau).signalTracksInvariantMass()<<endl;
      cout<<"# Signal Tracks "<<(*theCaloTau).signalTracks().size()<<endl;
      cout<<"# Isolation Tracks "<<(*theCaloTau).isolationTracks().size()<<endl;
      cout<<"Sum of Pt of the Tracks in isolation annulus around Lead Tk (GeV/c) "<<(*theCaloTau).isolationTracksPtSum()<<endl;
      cout<<"Sum of Et of the ECAL RecHits in other isolation annulus around Lead Tk (GeV) "<<(*theCaloTau).isolationECALhitsEtSum()<<endl;
      cout<<"Sum of Et of the HCAL hits inside a 3x3 calo. tower matrix centered on direction of propag. leading Track - ECAL inner surf. contact point (GeV) "<<(*theCaloTau).leadTrackHCAL3x3hitsEtSum()<<endl;
      cout<<"|DEta| between direction of propag. leading Track - ECAL inner surf. contact point and direction of highest Et hit among HCAL hits inside a 3x3 calo. tower matrix centered on direction of propag. leading Track - ECAL inner surf. contact point "<<(*theCaloTau).leadTrackHCAL3x3hottesthitDEta()<<endl;
    }
    i_CaloTau++;    
  }    
}
void CaloTauTest::endJob() { }


DEFINE_FWK_MODULE(CaloTauTest);
