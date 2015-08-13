#ifndef DataMixingTrackingParticleWorker_h
#define SimDataMixingTrackingParticleWorker_h

/** \class DataMixingTrackingParticleWorker
 *
 * DataMixingModule is the EDProducer subclass 
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the TrackingParticle information
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version October 2007
 *
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
//Data Formats
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"


#include <map>
#include <vector>
#include <string>



namespace edm
{
  class ModuleCallingContext;

  class DataMixingTrackingParticleWorker
    {
    public:

      DataMixingTrackingParticleWorker();

     /** standard constructor*/
      explicit DataMixingTrackingParticleWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC);

      /**Default destructor*/
      virtual ~DataMixingTrackingParticleWorker();

      virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c); // override?                            

      void putTrackingParticle(edm::Event &e) ;
      void addTrackingParticleSignals(const edm::Event &e); 
      void addTrackingParticlePileups(const int bcr, const edm::EventPrincipal*,unsigned int EventId,
                          ModuleCallingContext const*);

    private:
      // data specifiers

      edm::InputTag TrackingParticlecollectionSig_ ; // primary name given to collection of TrackingParticles
      edm::InputTag TrackingParticleLabelSig_ ;           // secondary name given to collection of TrackingParticles
      edm::InputTag TrackingParticlePileInputTag_ ;    // InputTag for pileup tracks
      std::string TrackingParticleCollectionDM_  ; // secondary name to be given to new TrackingParticle

      edm::InputTag StripLinkPileInputTag_;
      edm::InputTag PixelLinkPileInputTag_;
      edm::InputTag DTLinkPileInputTag_;
      edm::InputTag RPCLinkPileInputTag_;
      edm::InputTag CSCWireLinkPileInputTag_;
      edm::InputTag CSCStripLinkPileInputTag_;

      std::string StripLinkCollectionDM_;
      std::string PixelLinkCollectionDM_;
      std::string DTLinkCollectionDM_;
      std::string RPCLinkCollectionDM_;
      std::string CSCWireLinkCollectionDM_;
      std::string CSCStripLinkCollectionDM_;

      edm::EDGetTokenT<std::vector<TrackingParticle> >TrackSigToken_ ;  // Token to retrieve information  
      edm::EDGetTokenT<std::vector<TrackingParticle> >TrackPileToken_ ;  // Token to retrieve information  
      edm::EDGetTokenT<std::vector<TrackingVertex> >VtxSigToken_ ;  // Token to retrieve information  
      edm::EDGetTokenT<std::vector<TrackingVertex> >VtxPileToken_ ;  // Token to retrieve information  

      edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > StripLinkSigToken_;
      edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > StripLinkPileToken_;
      edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > PixelLinkSigToken_;
      edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > PixelLinkPileToken_;
      edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > CSCWireLinkSigToken_;
      edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > CSCWireLinkPileToken_;
      edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > CSCStripLinkSigToken_;
      edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > CSCStripLinkPileToken_;
      edm::EDGetTokenT< MuonDigiCollection<DTLayerId, DTDigiSimLink> > DTLinkSigToken_;
      edm::EDGetTokenT< MuonDigiCollection<DTLayerId, DTDigiSimLink> > DTLinkPileToken_;
      edm::EDGetTokenT<edm::DetSetVector<RPCDigiSimLink> > RPCLinkSigToken_;
      edm::EDGetTokenT<edm::DetSetVector<RPCDigiSimLink> > RPCLinkPileToken_;

      // 

      std::auto_ptr<std::vector<TrackingParticle>> NewTrackList_;
      std::auto_ptr<std::vector<TrackingVertex>> NewVertexList_;
      std::vector<TrackingVertex> TempVertexList_;

      std::unique_ptr<edm::DetSetVector<StripDigiSimLink> >          NewStripLinkList_;
      std::unique_ptr<edm::DetSetVector<PixelDigiSimLink> >          NewPixelLinkList_;
      std::unique_ptr< MuonDigiCollection<DTLayerId,DTDigiSimLink> > NewDTLinkList_;
      std::unique_ptr< edm::DetSetVector<RPCDigiSimLink> >           NewRPCLinkList_;
      std::unique_ptr< edm::DetSetVector<StripDigiSimLink> >         NewCSCWireLinkList_;
      std::unique_ptr< edm::DetSetVector<StripDigiSimLink> >         NewCSCStripLinkList_;

      TrackingParticleRefProd TrackListRef_ ;
      TrackingVertexRefProd VertexListRef_ ;


    };
}//edm

#endif
