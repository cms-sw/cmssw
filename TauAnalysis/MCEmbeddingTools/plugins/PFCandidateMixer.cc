// -*- C++ -*-
//
// Package:    PFCandidateMixer
// Class:      PFCandidateMixer
// 
/**\class PFCandidateMixer PFCandidateMixer.cc MyAna/PFCandidateMixer/src/PFCandidateMixer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Dec  9 16:14:56 CET 2009
// $Id: PFCandidateMixer.cc,v 1.6 2012/05/17 23:35:04 aburgmei Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <DataFormats/Common/interface/ValueMap.h>
#include <DataFormats/RecoCandidate/interface/RecoCandidate.h>
#include <DataFormats/Candidate/interface/CompositeRefCandidate.h>
#include <DataFormats/MuonReco/interface/Muon.h>

#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h>

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"

//
// class decleration
//

class PFCandidateMixer : public edm::EDProducer {
   public:
      explicit PFCandidateMixer(const edm::ParameterSet&);
      ~PFCandidateMixer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      void mix(edm::Event& iEvent, const edm::Handle<reco::TrackCollection>& trackCol, const edm::Handle<reco::MuonCollection>& muonCol, const edm::Handle<reco::GsfElectronCollection>& electronCol, const reco::PFCandidateCollection& pfIn1, const reco::PFCandidateCollection& pfIn2);
     
      edm::InputTag _col1;
      edm::InputTag _col2;

      edm::InputTag _trackCol;
      edm::InputTag _muonCol;
      edm::InputTag _electronCol;
      // ----------member data ---------------------------
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
PFCandidateMixer::PFCandidateMixer(const edm::ParameterSet& iConfig):
   _col1(iConfig.getUntrackedParameter<edm::InputTag>("col1") ),
   _col2(iConfig.getUntrackedParameter<edm::InputTag>("col2") ),
   _trackCol(iConfig.getUntrackedParameter<edm::InputTag>("trackCol") ),
   _muonCol(iConfig.getUntrackedParameter<edm::InputTag>("muons") ),
   _electronCol(iConfig.getUntrackedParameter<edm::InputTag>("gsfElectrons"))
{
   produces< std::vector< reco::PFCandidate >  >(); 

   if(!_electronCol.label().empty())
      produces< edm::ValueMap<edm::Ptr<reco::PFCandidate> > >("electrons"); 
#if 0
   if(!_muonCol.label().empty())
      produces< edm::ValueMap<edm::Ptr<reco::PFCandidate> > >("muons"); 
#endif
}

PFCandidateMixer::~PFCandidateMixer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PFCandidateMixer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   const bool muonColGiven = !_muonCol.label().empty();
   const bool electronColGiven = !_electronCol.label().empty();

   edm::Handle< std::vector<reco::Track> > trackCol;
   edm::Handle<reco::MuonCollection> muonCol;
   edm::Handle<reco::GsfElectronCollection> electronCol;
   iEvent.getByLabel( _trackCol, trackCol);

   if(muonColGiven)
      if(!iEvent.getByLabel(_muonCol, muonCol))
         throw cms::Exception("PFCandidateMixer") << "Muon Collection not found!";
   if(electronColGiven)
      if(!iEvent.getByLabel(_electronCol, electronCol))
         throw cms::Exception("PFCandidateMixer") << "GsfElectron Collection not found!";

   edm::Handle<reco::PFCandidateCollection> pfIn1;
   edm::Handle<reco::PFCandidateCollection> pfIn2;
   iEvent.getByLabel(_col1, pfIn1);
   iEvent.getByLabel(_col2, pfIn2);

   mix(iEvent, trackCol, muonCol, electronCol, *pfIn1, *pfIn2);
}

void PFCandidateMixer::mix(edm::Event& iEvent, const edm::Handle<reco::TrackCollection>& trackCol, const edm::Handle<reco::MuonCollection>& muonCol, const edm::Handle<reco::GsfElectronCollection>& electronCol, const reco::PFCandidateCollection& pfIn1, const reco::PFCandidateCollection& pfIn2)
{
   using namespace edm;
   using namespace reco;

   std::vector<const reco::PFCandidateCollection*> colVec;
   
   colVec.push_back(&pfIn1);
   colVec.push_back(&pfIn2);

   std::auto_ptr<std::vector< reco::PFCandidate > > pOut(new std::vector< reco::PFCandidate  > );
   
   std::vector<const reco::PFCandidateCollection*>::iterator itCol= colVec.begin();
   std::vector<const reco::PFCandidateCollection*>::iterator itColE= colVec.end();

   std::map<reco::GsfElectronRef, reco::PFCandidatePtr> electronCandidateMap; 

   int iCol = 0;
   for (;itCol!=itColE; ++itCol){
     PFCandidateConstIterator it = (*itCol)->begin();
     PFCandidateConstIterator itE = (*itCol)->end();
     for (;it!=itE;++it) {
      edm::Ptr<reco::PFCandidate> candPtr(*itCol, it - (*itCol)->begin());
      reco::PFCandidate cand(*it);
      //reco::PFCandidate cand(candPtr); // This breaks the output module, I'm not sure why
       
       //const bool isphoton   = cand.particleId() == reco::PFCandidate::gamma && cand.mva_nothing_gamma()>0.;
       const bool iselectron = cand.particleId() == reco::PFCandidate::e;
       //const bool hasNonNullMuonRef  = cand.muonRef().isNonnull();

       // if it is an electron. Find the GsfElectron with the same GsfTrack
       if (electronCol.isValid() && iselectron) {
         const reco::GsfTrackRef& gsfTrackRef(cand.gsfTrackRef());
         GsfElectronEqual myEqual(gsfTrackRef);
         std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(electronCol->begin(), electronCol->end(), myEqual);
         if(itcheck==electronCol->end())
           throw cms::Exception("PFCandidateMixer") << "GsfElectron for candidate not found!";

         reco::GsfElectronRef electronRef(electronCol, itcheck - electronCol->begin());
         cand.setGsfElectronRef(electronRef);
         cand.setSuperClusterRef(electronRef->superCluster());
         electronCandidateMap[electronRef] = candPtr;
       }

       // TODO: Do the same for muons -- see PFLinker.cc, we just don't have the MuonToMuonMap available.
       // But it might work with the pure muons collections as well.

       size_t i = 0;
       bool found = false;
       double minDR = 9999.;
       int iMinDr = -1;
       if (it->trackRef().isNonnull()) {
         for ( i = 0 ; i < trackCol->size(); ++i){
           if ( reco::deltaR( *(it->trackRef()), (*trackCol)[i] )<0.001 ) {
                found = true;
                break; 
           }
           double dr = reco::deltaR( *(it->trackRef()), (*trackCol)[i] );
           if ( dr < minDR) {
              iMinDr = i;
              minDR = dr;
           } 
         } 
       } 
       if ( found ){ // ref was found, overwrite in PFCand
         reco::TrackRef trref(trackCol,i);
         cand.setTrackRef(trref);
         //std::cout << " YY track ok"<<std::endl;

       } else { // keep orginall ref
         if (it->trackRef().isNonnull()) {
           std::cout << " XXXXXXXXXXX track not found " 
                 << " col " << iCol
                 << " ch " << it->charge()
                 << " id " << it->pdgId() 
                 << " pt " << it->pt() 
                 << " track: eta " << it->trackRef()->eta()
                 << " pt:  " << it->trackRef()->pt()
                 << " charge:  " << it->trackRef()->charge()
                 <<  std::endl;
           std::cout << " minDR=" << minDR << std::endl; 
           if ( iMinDr >= 0 ) {
                std::cout 
                     << " closest track pt=" << (*trackCol)[iMinDr].pt()
                     << " ch=" << (*trackCol)[iMinDr].charge()
                     <<  std::endl; 
           } 
           edm::Provenance prov=iEvent.getProvenance(it->trackRef().id());
           edm::InputTag tag(prov.moduleLabel(),  prov.productInstanceName(),   prov.processName());
           std::cout << " trackref in PFCand came from: "   << tag.encode() << std::endl;
         }
       }
       pOut->push_back(cand);
     }
     ++iCol;
   }

   edm::OrphanHandle<reco::PFCandidateCollection> newColl = iEvent.put(pOut);

   // Now fixup the references and write the valuemap
   if(electronCol.isValid())
   {
      std::vector<reco::PFCandidatePtr> values(electronCol->size());
      for(unsigned int i = 0; i < electronCol->size(); ++i)
      {
         edm::Ref<reco::GsfElectronCollection> objRef(electronCol, i);
         std::map<reco::GsfElectronRef, reco::PFCandidatePtr>::const_iterator iter = electronCandidateMap.find(objRef);

         reco::PFCandidatePtr candPtr;
         if(iter != electronCandidateMap.end())
            candPtr = reco::PFCandidatePtr(newColl, iter->second.key());
         values[i] = candPtr;
      }

      std::auto_ptr<edm::ValueMap<reco::PFCandidatePtr> > pfMap_p(new edm::ValueMap<reco::PFCandidatePtr>());
      edm::ValueMap<reco::PFCandidatePtr>::Filler filler(*pfMap_p);
      filler.insert(electronCol, values.begin(), values.end());
      filler.fill();
      iEvent.put(pfMap_p, "electrons");
   }

   // TODO: Do the same for muons
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFCandidateMixer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFCandidateMixer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFCandidateMixer);
