// Producer for validation histograms for PFJet objects
// J. Weng, based on F. Ratnikov, Sept. 7, 2006

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/JetReco/interface/PFJet.h"


#include "Validation/RecoJets/src/PFJetTester.h"

using namespace edm;
using namespace reco;
using namespace std;

PFJetTester::PFJetTester(const edm::ParameterSet& iConfig)
  : mInputCollection (iConfig.getParameter<edm::InputTag>( "src" )),
    mOutputFile (iConfig.getUntrackedParameter<string>("outputFile", ""))
{
  //**** @@@
  mEta = mPhi = mE = mP = mPt = mMass = mConstituents
    = mEtaFirst = mPhiFirst = mEFirst = mPtFirst 
    =mChargedHadronEnergy
    = mNeutralHadronEnergy = mChargedEmEnergy = mChargedMuEnergy
    = mNeutralEmEnergy =  mChargedMultiplicity = mNeutralMultiplicity
    = mMuonMultiplicity = 0;
  
  DaqMonitorBEInterface* dbe = &*edm::Service<DaqMonitorBEInterface>();
  if (dbe) {
    dbe->setCurrentFolder("PFJetTask_" + mInputCollection.label());
    mEta = dbe->book1D("Eta", "Eta", 100, -5, 5); 
    mPhi = dbe->book1D("Phi", "Phi", 70, -3.5, 3.5); 
    mE = dbe->book1D("E", "E", 100, 0, 500); 
    mP = dbe->book1D("P", "P", 100, 0, 500); 
    mPt = dbe->book1D("Pt", "Pt", 100, 0, 50); 
    mMass = dbe->book1D("Mass", "Mass", 100, 0, 25); 
    mConstituents = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100); 
    //
    mEtaFirst = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5); 
    mPhiFirst = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5); 
    mEFirst = dbe->book1D("EFirst", "EFirst", 100, 0, 1000); 
    mPtFirst = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500); 
    //
    mChargedHadronEnergy = dbe->book1D("mChargedHadronEnergy", "mChargedHadronEnergy", 100, 0, 100); 
    mNeutralHadronEnergy = dbe->book1D("mNeutralHadronEnergy", "mNeutralHadronEnergy", 100, 0, 100);  
    mChargedEmEnergy= dbe->book1D("mChargedEmEnergy ", "mChargedEmEnergy ", 100, 0, 100); 
    mChargedMuEnergy = dbe->book1D("mChargedMuEnergy", "mChargedMuEnergy", 100, 0, 100); 
    mNeutralEmEnergy= dbe->book1D("mNeutralEmEnergy", "mNeutralEmEnergy", 100, 0, 100);   
    mChargedMultiplicity= dbe->book1D("mChargedMultiplicity ", "mChargedMultiplicity ", 100, 0, 100);     
    mNeutralMultiplicity = dbe->book1D(" mNeutralMultiplicity", "mNeutralMultiplicity", 100, 0, 100);
    mMuonMultiplicity= dbe->book1D("mMuonMultiplicity", "mMuonMultiplicity", 100, 0, 100);
  }
  if (mOutputFile.empty ()) {
    LogInfo("OutputInfo") << " PFJet histograms will NOT be saved";
  } 
  else {
    LogInfo("OutputInfo") << " PFJethistograms will be saved to file:" << mOutputFile;
  }
}
   
PFJetTester::~PFJetTester()
{
}

void PFJetTester::beginJob(const edm::EventSetup& c){
}

void PFJetTester::endJob() {
  if (!mOutputFile.empty() && &*edm::Service<DaqMonitorBEInterface>()) edm::Service<DaqMonitorBEInterface>()->save (mOutputFile);
}


void PFJetTester::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{
  Handle<PFJetCollection> collection;
  mEvent.getByLabel(mInputCollection, collection);
  //  cout <<"!!!!!!!!!!! " <<mInputCollection << " " <<collection->size() <<endl;
  PFJetCollection::const_iterator jet = collection->begin ();
  for (; jet != collection->end (); jet++) {
    if (mEta) mEta->Fill (jet->eta());
    if (mPhi) mPhi->Fill (jet->phi());
    if (mE) mE->Fill (jet->energy());
    if (mP) mP->Fill (jet->p());
    if (mPt) mPt->Fill (jet->pt());
    if (mMass) mMass->Fill (jet->mass());
    if (mConstituents) mConstituents->Fill (jet->nConstituents());
    if (jet == collection->begin ()) { // first jet
      if (mEtaFirst) mEtaFirst->Fill (jet->eta());
      if (mPhiFirst) mPhiFirst->Fill (jet->phi());
      if (mEFirst) mEFirst->Fill (jet->energy());
      if (mPtFirst) mPtFirst->Fill (jet->pt());
    }
    if (mChargedHadronEnergy)  mChargedHadronEnergy->Fill (jet->chargedHadronEnergy());
    if (mNeutralHadronEnergy)  mNeutralHadronEnergy->Fill (jet->neutralHadronEnergy());
    if (mChargedEmEnergy) mChargedEmEnergy->Fill(jet->chargedEmEnergy());
    if (mChargedMuEnergy) mChargedMuEnergy->Fill (jet->chargedMuEnergy ());
    if (mNeutralEmEnergy) mNeutralEmEnergy->Fill(jet->neutralEmEnergy()); 
    if (mChargedMultiplicity ) mChargedMultiplicity->Fill(jet->chargedMultiplicity());
    if (mNeutralMultiplicity ) mNeutralMultiplicity->Fill(jet->neutralMultiplicity());
    if (mMuonMultiplicity )mMuonMultiplicity->Fill (jet-> muonMultiplicity());
  }
}

