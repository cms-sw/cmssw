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

      edm::EDGetTokenT<std::vector<TrackingParticle> >TrackSigToken_ ;  // Token to retrieve information  
      edm::EDGetTokenT<std::vector<TrackingParticle> >TrackPileToken_ ;  // Token to retrieve information  
      edm::EDGetTokenT<std::vector<TrackingVertex> >VtxSigToken_ ;  // Token to retrieve information  
      edm::EDGetTokenT<std::vector<TrackingVertex> >VtxPileToken_ ;  // Token to retrieve information  

      // 

      std::auto_ptr<std::vector<TrackingParticle>> NewTrackList_;
      std::auto_ptr<std::vector<TrackingVertex>> NewVertexList_;
      std::vector<TrackingVertex> TempVertexList_;

      TrackingParticleRefProd TrackListRef_ ;
      TrackingVertexRefProd VertexListRef_ ;


    };
}//edm

#endif
