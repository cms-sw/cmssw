// -*- C++ -*-
//
// Package:    SimTracker/TrackAssociatorProducers
// Class:      QuickTrackAssociatorByHitsProducer
// 
/**\class QuickTrackAssociatorByHitsProducer QuickTrackAssociatorByHitsProducer.cc SimTracker/TrackAssociatorProducers/plugins/QuickTrackAssociatorByHitsProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Mon, 05 Jan 2015 16:33:55 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "QuickTrackAssociatorByHitsImpl.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

//
// class declaration
//
namespace {
}

class QuickTrackAssociatorByHitsProducer : public edm::global::EDProducer<> {
   public:
      explicit QuickTrackAssociatorByHitsProducer(const edm::ParameterSet&);
      ~QuickTrackAssociatorByHitsProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      virtual void endJob() override;
      edm::ParameterSet makeHitAssociatorParameters(const edm::ParameterSet&);
      
      // ----------member data ---------------------------
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  edm::EDGetTokenT<ClusterTPAssociationList> cluster2TPToken_;
  double qualitySimToReco_;
  double puritySimToReco_;
  double cutRecoToSim_;
  QuickTrackAssociatorByHitsImpl::SimToRecoDenomType simToRecoDenominator_;
  bool threeHitTracksAreSpecial_;
  bool useClusterTPAssociation_;
  bool absoluteNumberOfHits_;

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
QuickTrackAssociatorByHitsProducer::QuickTrackAssociatorByHitsProducer(const edm::ParameterSet& iConfig):
  trackerHitAssociatorConfig_(makeHitAssociatorParameters(iConfig), consumesCollector()),
  qualitySimToReco_( iConfig.getParameter<double>( "Quality_SimToReco" ) ),
  puritySimToReco_( iConfig.getParameter<double>( "Purity_SimToReco" ) ),
  cutRecoToSim_( iConfig.getParameter<double>( "Cut_RecoToSim" ) ),
  threeHitTracksAreSpecial_( iConfig.getParameter<bool>( "ThreeHitTracksAreSpecial" ) ),
  useClusterTPAssociation_( iConfig.getParameter<bool>( "useClusterTPAssociation" ) ),
  absoluteNumberOfHits_( iConfig.getParameter<bool>( "AbsoluteNumberOfHits" ) )
{
  
  //
  // Check whether the denominator when working out the percentage of shared hits should
  // be the number of simulated hits or the number of reconstructed hits.
  //
  std::string denominatorString=iConfig.getParameter<std::string>("SimToRecoDenominator");
  if( denominatorString=="sim" ) simToRecoDenominator_=QuickTrackAssociatorByHitsImpl::denomsim;
  else if( denominatorString=="reco" ) simToRecoDenominator_=QuickTrackAssociatorByHitsImpl::denomreco;
  else throw cms::Exception( "QuickTrackAssociatorByHitsImpl" ) << "SimToRecoDenominator not specified as sim or reco";
  
  //
  // Do some checks on whether UseGrouped or UseSplitting have been set. They're not used
  // unlike the standard TrackAssociatorByHits so show a warning.
  //
  bool useGrouped, useSplitting;
  if( iConfig.exists("UseGrouped") ) useGrouped=iConfig.getParameter<bool>("UseGrouped");
  else useGrouped=true;
  
  if( iConfig.exists("UseSplitting") ) useSplitting=iConfig.getParameter<bool>("UseSplitting");
  else useSplitting=true;
  
  // This associator works as though both UseGrouped and UseSplitting were set to true, so show a
  // warning if this isn't the case.
  if( !(useGrouped && useSplitting) )
    {
      LogDebug("QuickTrackAssociatorByHitsImpl") << "UseGrouped and/or UseSplitting has been set to false, but this associator ignores that setting.";
    }
  
  //register your products
  produces<reco::TrackToTrackingParticleAssociator>();  

  if(useClusterTPAssociation_) {
    cluster2TPToken_ = consumes<ClusterTPAssociationList>(iConfig.getParameter < edm::InputTag > ("cluster2TPSrc"));
  }

}


QuickTrackAssociatorByHitsProducer::~QuickTrackAssociatorByHitsProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//
  
// Set up the parameter set for the hit associator
edm::ParameterSet
QuickTrackAssociatorByHitsProducer::makeHitAssociatorParameters(const edm::ParameterSet& iConfig) {
  edm::ParameterSet hitAssociatorParameters;
  hitAssociatorParameters.addParameter<bool>( "associatePixel", iConfig.getParameter<bool>("associatePixel") );
  hitAssociatorParameters.addParameter<bool>( "associateStrip", iConfig.getParameter<bool>("associateStrip") );
  // This is the important one, it stops the hit associator searching through the list of sim hits.
  // I only want to use the hit associator methods that work on the hit IDs (i.e. the uint32_t trackId
  // and the EncodedEventId eventId) so I'm not interested in matching that to the PSimHit objects.
  hitAssociatorParameters.addParameter<bool>("associateRecoTracks",true);
  // add these new ones to allow redirection of inputs:
  hitAssociatorParameters.addParameter<edm::InputTag>( "pixelSimLinkSrc", iConfig.getParameter<edm::InputTag>("pixelSimLinkSrc") );
  hitAssociatorParameters.addParameter<edm::InputTag>( "stripSimLinkSrc", iConfig.getParameter<edm::InputTag>("stripSimLinkSrc") );

  return hitAssociatorParameters;
}

// ------------ method called to produce the data  ------------
void
QuickTrackAssociatorByHitsProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{
   using namespace edm;

   const ClusterTPAssociationList *clusterAssoc = nullptr;
   std::unique_ptr<TrackerHitAssociator> trackAssoc;
   if(useClusterTPAssociation_)  {
     edm::Handle<ClusterTPAssociationList> clusterAssocHandle;
     iEvent.getByToken(cluster2TPToken_,clusterAssocHandle);

     if(clusterAssocHandle.isValid()) {
       clusterAssoc = clusterAssocHandle.product();
     } else {
       edm::LogInfo( "TrackAssociator" ) << "ClusterTPAssociationList not found. Using DigiSimLink based associator";
     }
   }
   if(not clusterAssoc) {
     // If control got this far then either useClusterTPAssociation_ was false or getting the cluster
     // to TrackingParticle association from the event failed. Either way I need to create a hit associator.
     trackAssoc = std::make_unique<TrackerHitAssociator>(iEvent, trackerHitAssociatorConfig_);
   }

   auto impl = std::make_unique<QuickTrackAssociatorByHitsImpl>(iEvent.productGetter(),
                                                                std::move(trackAssoc),
                                                                clusterAssoc,
                                                                absoluteNumberOfHits_,
                                                                qualitySimToReco_,
                                                                puritySimToReco_,
                                                                cutRecoToSim_,
                                                                threeHitTracksAreSpecial_,
                                                                simToRecoDenominator_);

   auto toPut = std::make_unique<reco::TrackToTrackingParticleAssociator>(std::move(impl));
   iEvent.put(std::move(toPut));
}

// ------------ method called once each job just before starting event loop  ------------
void 
QuickTrackAssociatorByHitsProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
QuickTrackAssociatorByHitsProducer::endJob() {
}

 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
QuickTrackAssociatorByHitsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(QuickTrackAssociatorByHitsProducer);
