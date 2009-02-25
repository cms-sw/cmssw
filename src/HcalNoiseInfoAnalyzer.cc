//
// HcalNoiseInfoProducer.cc
//
//   description: Implementation of skeleton analyzer for the HCAL noise information
//
//   author: J.P. Chou, Brown
//
//

#include "RecoMET/METAnalyzers/interface/HcalNoiseInfoAnalyzer.h"
#include "DataFormats/METReco/interface/HcalNoiseRBXArray.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "TFile.h"
#include "TH1D.h"

using namespace reco;
  
//
// constructors and destructor
//
  
HcalNoiseInfoAnalyzer::HcalNoiseInfoAnalyzer(const edm::ParameterSet& iConfig)
{
  // set parameters
  rbxCollName_    = iConfig.getParameter<std::string>("rbxCollName");
  rootHistFilename_ = iConfig.getParameter<std::string>("rootHistFilename");
}
  
  
HcalNoiseInfoAnalyzer::~HcalNoiseInfoAnalyzer()
{
}
  
  
//
// member functions
//
  
// ------------ method called to for each event  ------------
void
HcalNoiseInfoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
    
  Handle<HcalNoiseRBXCollection> handle;
  iEvent.getByLabel(rbxCollName_,handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find HcalNoiseRBXCollection named " << rbxCollName_ << ".\n";
    return;
  }
    
  // loop over the RBXs
  for(HcalNoiseRBXCollection::const_iterator rit=handle->begin(); rit!=handle->end(); ++rit) {
    // get the rbx
    HcalNoiseRBX rbx=(*rit);
      
    // get the highest rechit energy HPD in the RBX
    HcalNoiseHPD maxhpd=(*rbx.maxHPD());
      
    // loop over the HPDs in each RBX
    int hpdcntr=0;
    for(HcalNoiseHPDArray::const_iterator hit=rbx.beginHPD(); hit!=rbx.endHPD(); ++hit) {
      HcalNoiseHPD hpd=(*hit);
	
      // loop over highest energy Digi
      int cntr=0;
      for(DigiArray::const_iterator dit=hpd.beginBigDigi(); dit!=hpd.endBigDigi(); ++dit) {
	double fC=(*dit);
	 
      }

      // loop over the 5 highest energy rechits in the HPD
      // This is here for debugging purposes only: probably won't be available later
      EnergySortedHBHERecHits rechits=hpd.recHits();
      for(EnergySortedHBHERecHits::const_iterator it=rechits.begin(); it!=rechits.end(); ++it) {

      }

      hNHPDHits_->Fill(hpd.numHitsAboveThreshold());
    }

    hNHPDMaxHits_->Fill(maxhpd.numHitsAboveThreshold());
  }


  return;
}


// ------------ method called once each job just before starting event loop  ------------
void 
HcalNoiseInfoAnalyzer::beginJob(const edm::EventSetup&)
{
  // book histograms
  rootfile_ = new TFile(rootHistFilename_.c_str(), "RECREATE");

  hNHPDHits_ = new TH1D("hNHPDHits","Number of RecHits in an HPD",19,0,19);
  hNHPDMaxHits_ = new TH1D("hNHPDMaxHits","Number of RecHits in the highest energy HPD in the RBX",19,0,19);

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalNoiseInfoAnalyzer::endJob() {

  // write histograms
  rootfile_->cd();
  hNHPDHits_->Write();
  hNHPDMaxHits_->Write();
  rootfile_->Close();
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalNoiseInfoAnalyzer);
