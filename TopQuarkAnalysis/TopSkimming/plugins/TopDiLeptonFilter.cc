// Original Author:  Dmytro Kovalskyi, UCSB
// $Id: TopDiLeptonFilter.cc,v 1.4 2007/12/10 12:40:00 lowette Exp $
#include "TopQuarkAnalysis/TopSkimming/plugins/TopDiLeptonFilter.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

TopDiLeptonFilter::TopDiLeptonFilter(const edm::ParameterSet& iConfig)
{
   thePtThreshold        = iConfig.getParameter<double>("ptThreshold");
   theElectronCollection = iConfig.getParameter<edm::InputTag>("electronCollection");
   theMuonCollection     = iConfig.getParameter<edm::InputTag>("muonCollection");
}

bool TopDiLeptonFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(theMuonCollection,muons);
   edm::Handle<reco::GsfElectronCollection> electrons;
   iEvent.getByLabel(theElectronCollection,electrons);
   
   unsigned int nMuons = 0;
   unsigned int nElectrons = 0;
   for(reco::MuonCollection::const_iterator muon=muons->begin(); muon!=muons->end(); ++muon)
     if(muon->pt()>thePtThreshold) ++nMuons;
   for(reco::GsfElectronCollection::const_iterator electron=electrons->begin(); electron!=electrons->end(); ++electron)
     if(electron->pt()>thePtThreshold) ++nElectrons;
   
   if (nMuons+nElectrons>1)
     return true;
   else
     return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TopDiLeptonFilter);
   

