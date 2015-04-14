//
// NoiseRates.cc
//
//   description: Calculation for single particle response corrections
//
//   author: J.P. Chou, Brown
//
//

#include "Validation/HcalRecHits/interface/NoiseRates.h"
#include "FWCore/Utilities/interface/EDMException.h"

//
// constructors and destructor
//

NoiseRates::NoiseRates(const edm::ParameterSet& iConfig)
{

  // DQM ROOT output
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile","myfile.root");

  // set parameters
  rbxCollName_   = iConfig.getUntrackedParameter<edm::InputTag>("rbxCollName");
  minRBXEnergy_  = iConfig.getUntrackedParameter<double>("minRBXEnergy");
  minHitEnergy_  = iConfig.getUntrackedParameter<double>("minHitEnergy");

  tok_rbx_ = consumes<reco::HcalNoiseRBXCollection>(rbxCollName_);

  useAllHistos_  = iConfig.getUntrackedParameter<bool>("useAllHistos", false);

}
  
  
NoiseRates::~NoiseRates()
{
}
  
  
//
// member functions
//

void NoiseRates::bookHistograms(DQMStore::IBooker &ib, edm::Run const &run, edm::EventSetup const &es)
{

  ib.setCurrentFolder("NoiseRatesV/NoiseRatesTask");

  // book histograms
  Char_t histo[100];

  //Lumi block is not drawn; the rest are
  if (useAllHistos_){
    sprintf  (histo, "hLumiBlockCount" );
    hLumiBlockCount_ = ib.book1D(histo, histo, 1, -0.5, 0.5);
  }
  
  sprintf  (histo, "hRBXEnergy" );
  hRBXEnergy_ = ib.book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType1" );
  hRBXEnergyType1_ = ib.book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType2" );
  hRBXEnergyType2_ = ib.book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXEnergyType3" );
  hRBXEnergyType3_ = ib.book1D(histo, histo, 300, 0, 3000);

  sprintf  (histo, "hRBXNHits" );
  hRBXNHits_ = ib.book1D(histo, histo, 73,-0.5,72.5);

}
  
// ------------ method called to for each event  ------------
void
NoiseRates::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

  // get the lumi section
  int lumiSection = iEvent.luminosityBlock();
  lumiCountMap_[lumiSection]++;

  // get the RBX Noise collection
  edm::Handle<reco::HcalNoiseRBXCollection> handle;
  iEvent.getByToken(tok_rbx_,handle);
  if(!handle.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound)
      << " could not find HcalNoiseRBXCollection named " << rbxCollName_ << ".\n";
    return;
  }

  // loop over the RBXs and fill the histograms
  for(reco::HcalNoiseRBXCollection::const_iterator it=handle->begin(); it!=handle->end(); ++it) {
    const reco::HcalNoiseRBX &rbx=(*it);

    double energy = rbx.recHitEnergy(minHitEnergy_);

    int nhits = rbx.numRecHits(minHitEnergy_);

    if(energy < minRBXEnergy_) continue;

    hRBXEnergy_->Fill(energy);
    
    if      (nhits <= 9)  hRBXEnergyType1_->Fill(energy);
    else if (nhits <= 18) hRBXEnergyType2_->Fill(energy);
    else               	  hRBXEnergyType3_->Fill(energy);
    
    hRBXNHits_->Fill(nhits);
    
  }   // done looping over RBXs

}


//define this as a plug-in
DEFINE_FWK_MODULE(NoiseRates);
