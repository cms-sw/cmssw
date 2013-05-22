// -*- C++ -*-
//
// Package:    L1CaloClusterAnalyzer
// Class:      L1CaloClusterAnalyzer
//
/**\class L1CaloClusterAnalyzer L1CaloClusterAnalyzer.cc SLHCUpgradeSimulations/L1CaloClusterAnalyzer/src/L1CaloClusterAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Isobel Ojalvo
//         Created:  Mon Feb 13 05:35:01 CST 2012


#include "SLHCUpgradeSimulations/L1CaloTrigger/plugins/L1CaloClusterAnalyzer.h"

#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>


L1CaloClusterAnalyzer::L1CaloClusterAnalyzer(const edm::ParameterSet& iConfig):
  src_(iConfig.getParameter<edm::InputTag>("src")),
  electrons_(iConfig.getParameter<edm::InputTag>("electrons"))
{
   //now do what ever initialization is needed


  edm::Service<TFileService> fs;
  RRTree = fs->make<TTree>("RRTree","Tree containing RAW RECO info");

  //RRTree->Branch("coneEnergy",&coneE);
  RRTree->Branch("L1Pt",&centralPt);
  RRTree->Branch("RecoPt",&RecoPt);
  RRTree->Branch("RecoMatch",&RecoMatch);
  RRTree->Branch("ClusterPtMatch",&ClusterPtMatch);
  RRTree->Branch("CentralIso",&CentralIso);
  RRTree->Branch("TowerEnergy1",&TowerEnergy1);
  RRTree->Branch("TowerEnergy2",&TowerEnergy2);
  RRTree->Branch("TowerEnergy3",&TowerEnergy3);
  RRTree->Branch("TowerEnergy4",&TowerEnergy4);
  RRTree->Branch("Ring1E",&Ring1E);
  RRTree->Branch("Ring2E",&Ring2E);
  RRTree->Branch("Ring3E",&Ring3E);
  RRTree->Branch("Ring4E",&Ring4E);
  RRTree->Branch("ClusterEnergy",&ClusterEnergy);


}


L1CaloClusterAnalyzer::~L1CaloClusterAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void
L1CaloClusterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<l1slhc::L1CaloClusterCollection> clusters;
   iEvent.getByLabel(src_,clusters);

  edm::Handle<reco::GsfElectronCollection> electrons;
   
   bool gotRecoE = iEvent.getByLabel(electrons_,electrons);   
   
   for(unsigned int j=0;j<clusters->size();++j)
     {
       if(clusters->at(j).isCentral() && clusters->at(j).isEGamma() ){
	 coneE = clusters->at(j).isoEnergyEG();
	 centralPt = clusters->at(j).p4().pt();

	 CentralIso = (float) (clusters->at(j).LeadTowerE())/(clusters->at(j).Et());

	 TowerEnergy1 = clusters->at(j).LeadTowerE();
	 TowerEnergy2 = clusters->at(j).SecondTowerE();
	 TowerEnergy3 = clusters->at(j).ThirdTowerE();
	 TowerEnergy4 = clusters->at(j).FourthTowerE();
	 Ring1E = clusters->at(j).Ring1E();
	 Ring2E = clusters->at(j).Ring2E();
	 Ring3E = clusters->at(j).Ring3E();
	 Ring4E = clusters->at(j).Ring4E();

	 ClusterEnergy = clusters->at(j).Et();

	 //printf("CentralIso: %f\n",CentralIso);

     	 //	 printf("Number of Clusters = %i \n",clusters->at(j).isoClusters());

	 bool passID = false;
	 RecoPt = 0;

	 if(gotRecoE)

	   for( unsigned int i =1; i<electrons->size() && !passID; ++i){
	     if((electrons->at(i).dr04TkSumPt() + electrons->at(i).dr04EcalRecHitSumEt() + electrons->at(i).dr04HcalTowerSumEt())/(electrons->at(i).pt())<0.15)//
	       if(electrons->at(i).isEB()||electrons->at(i).isEE())
		 if(fabs(electrons->at(i).sigmaIetaIeta())<0.025)  //sigmaEtaEta_[type])
		   if(fabs(electrons->at(i).deltaEtaSuperClusterTrackAtVtx())<0.02)  //deltaEta_[type])
		     if(fabs(electrons->at(i).deltaPhiSuperClusterTrackAtVtx())<0.1)//deltaPhi_[type])
		       if(fabs(electrons->at(i).hcalOverEcal())<0.01)    //hoE_[type])
			 if((electrons->at(i).dr03TkSumPt()+electrons->at(i).dr03EcalRecHitSumEt()+electrons->at(i).dr03HcalDepth1TowerSumEt())/electrons->at(i).pt()<0.15)
			   if(ROOT::Math::VectorUtil::DeltaR(clusters->at(j).p4(),electrons->at(i).p4())<0.3) {
			     passID=true;
			     RecoPt = electrons->at(i).pt();
			   }
	   }

	   if(passID){
	     ClusterPtMatch = fabs((clusters->at(j).p4().pt()-RecoPt)/RecoPt);
	     RecoMatch = 1;
	   }
	   else{
	     ClusterPtMatch = -1;
	     RecoMatch = -1;
	   }

	   RRTree->Fill();
       }
     }

}


// ------------ method called once each job just before starting event loop  ------------
void
L1CaloClusterAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
L1CaloClusterAnalyzer::endJob()
{
}

// ------------ method called when starting to processes a run  ------------
void
L1CaloClusterAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
L1CaloClusterAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
L1CaloClusterAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
L1CaloClusterAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

void
L1CaloClusterAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


//define this as a plug-in
DEFINE_FWK_MODULE(L1CaloClusterAnalyzer);
