// -*- C++ -*-
//
// Package:    MonoPhotonSkimmer
// Class:      MonoPhotonSkimmer
//
/**\class MonoPhotonSkimmer MonoPhotonSkimmer.cc MonoPhotonSkimmer/MonoPhotonSkimmer/src/MonoPhotonSkimmer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jie Chen
//         Created:  Wed Nov 17 14:33:08 CST 2010
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <string>

//
// class declaration
//

class MonoPhotonSkimmer : public edm::EDFilter {
   public:
      explicit MonoPhotonSkimmer(const edm::ParameterSet&);
      ~MonoPhotonSkimmer();

   private:
      virtual void beginJob() override ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      // ----------member data ---------------------------
      edm::EDGetTokenT<reco::PhotonCollection> _phoToken;
      bool _selectEE;         //Do you want to select EE photons?
      //True enables this.

      double _ecalisoOffsetEB; //Photon Preselection has linearized cuts.
      double _ecalisoSlopeEB;  //slope * photonpt + offset is the isolation
      //threshold.  This is ECAL EB.

      double _hcalisoOffsetEB; //Linearized cut on HCAL towers, EB.
      double _hcalisoSlopeEB;

      double _hadoveremEB;     //Flat selection cut on HadOverEM.

      double _minPhoEtEB;      //Minimum Photon ET threshold, EB.

      double _trackisoOffsetEB;//Linearized cut on track isolation EB
      double _trackisoSlopeEB;

      double _etawidthEB;        //eta width for EB

      double _ecalisoOffsetEE; //As above, but separately set for EE.
      double _ecalisoSlopeEE;
      double _hcalisoOffsetEE;
      double _hcalisoSlopeEE;
      double _hadoveremEE;
      double _minPhoEtEE;
      double _trackisoOffsetEE;
      double _trackisoSlopeEE;
      double _etawidthEE;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MonoPhotonSkimmer::MonoPhotonSkimmer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  _phoToken = consumes<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("phoTag"));
   _selectEE = iConfig.getParameter<bool>("selectEE");

  _ecalisoOffsetEB = iConfig.getParameter<double>("ecalisoOffsetEB");
  _ecalisoSlopeEB = iConfig.getParameter<double>("ecalisoSlopeEB");

  _hcalisoOffsetEB = iConfig.getParameter<double>("hcalisoOffsetEB");
  _hcalisoSlopeEB  = iConfig.getParameter<double>("hcalisoSlopeEB");

  _hadoveremEB = iConfig.getParameter<double>("hadoveremEB");
  _minPhoEtEB = iConfig.getParameter<double>("minPhoEtEB");


  _trackisoOffsetEB= iConfig.getParameter<double>("trackIsoOffsetEB");
  _trackisoSlopeEB= iConfig.getParameter<double>("trackIsoSlopeEB");
  _etawidthEB=iConfig.getParameter<double>("etaWidthEB");

  _ecalisoOffsetEE = iConfig.getParameter<double>("ecalisoOffsetEE");
  _ecalisoSlopeEE = iConfig.getParameter<double>("ecalisoSlopeEE");

  _hcalisoOffsetEE = iConfig.getParameter<double>("hcalisoOffsetEE");
  _hcalisoSlopeEE  = iConfig.getParameter<double>("hcalisoSlopeEE");

  _hadoveremEE = iConfig.getParameter<double>("hadoveremEE");
  _minPhoEtEE = iConfig.getParameter<double>("minPhoEtEE");


  _trackisoOffsetEE= iConfig.getParameter<double>("trackIsoOffsetEE");
  _trackisoSlopeEE= iConfig.getParameter<double>("trackIsoSlopeEE");

  _etawidthEE= iConfig.getParameter<double>("etaWidthEE");

}


MonoPhotonSkimmer::~MonoPhotonSkimmer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
MonoPhotonSkimmer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  Handle<reco::PhotonCollection> photonColl;
  iEvent.getByToken(_phoToken, photonColl);
  const reco::PhotonCollection *photons = photonColl.product();
  //Iterate over photon collection.
//  std::vector<reco::Photon> PreselPhotons;
  int PreselPhotons=0;
  reco::PhotonCollection::const_iterator pho;
  for (pho = (*photons).begin(); pho!= (*photons).end(); pho++){
    if (!pho->isEB() && !_selectEE) continue;

    double ecalisocut = 0;
    double hcalisocut = 0;
    double hadoverem  = 0;
    double minphoet   = 0;
    double trackiso   = 0;
    double etawidth   = 0;
    if (pho->isEB()){
      ecalisocut = _ecalisoOffsetEB + _ecalisoSlopeEB * pho->pt();
      hcalisocut = _hcalisoOffsetEB + _hcalisoSlopeEB * pho->pt();
      hadoverem  = _hadoveremEB;
      minphoet   = _minPhoEtEB;
      trackiso   = _trackisoOffsetEB + _trackisoSlopeEB * pho->pt();
      etawidth   = _etawidthEB;
    }
    else{
      ecalisocut = _ecalisoOffsetEE + _ecalisoSlopeEE * pho->pt();
      hcalisocut = _hcalisoOffsetEE + _hcalisoSlopeEE * pho->pt();
      hadoverem  = _hadoveremEE;
      minphoet   = _minPhoEtEE;
      trackiso   = _trackisoOffsetEE + _trackisoSlopeEE * pho->pt();
      etawidth   = _etawidthEE;
   }

    if (pho->ecalRecHitSumEtConeDR04() < ecalisocut
        && pho->hcalTowerSumEtConeDR04() < hcalisocut
        && pho->hadronicOverEm() < hadoverem
        && pho->pt() > minphoet
        && pho->trkSumPtHollowConeDR04()<trackiso
        && pho->sigmaIetaIeta() <etawidth
       ) PreselPhotons++;

  }//Loop over Photons
  if (PreselPhotons > 0 ) return true;
  return false;
}

// ------------ method called once each job just before starting event loop  ------------
void
MonoPhotonSkimmer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
MonoPhotonSkimmer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MonoPhotonSkimmer);
