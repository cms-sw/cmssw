#ifndef MixingModule_h
#define SimMixingModule_h

/** \class MixingModule
 *
 * MixingModule is the EDProducer subclass 
 * which fills the CrossingFrame object to allow to add
 * pileup events in digitisations
 *
 * \author Ursula Berthon, LLR Palaiseau
 *
 * \version   1st Version June 2005
 * \version   2nd Version Sep 2005

 *
 ************************************************************/
#include "Mixing/Base/interface/BMixingModule.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Selector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

#include <vector>
#include <string>


namespace edm
{
  class MixingModule : public BMixingModule
    {
    public:

      /** standard constructor*/
      explicit MixingModule(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~MixingModule();

      virtual void beginJob(edm::EventSetup const&iSetup);

      // limits for tof to be considered for trackers
        static const int lowTrackTof; //nsec
        static const int highTrackTof;
        static const int limHighLowTof;
 
    private:
      virtual void put(edm::Event &e) ;
      virtual void createnewEDProduct();
      virtual void addSignals(const edm::Event &e); 
      virtual void addPileups(const int bcr, edm::Event*,unsigned int EventId);
      virtual void getSubdetectorNames();

      // internally used information : subdetectors present in input
      std::vector<std::string> simHitSubdetectors_;
      std::vector<std::string> caloSubdetectors_;
      // to distinguish simHitSubdetectors for tracker/non-tracker
      // this is necessary to know which ones have to be checked for ToF
      std::vector<std::string> trackerHighLowPids_;
      std::vector<std::string> nonTrackerPids_;

      // in this map we put the CrossingFrame objects that were created per event
      std::map<std::string,CrossingFrame<PSimHit> * > cfSimHits_;
      std::map<std::string,CrossingFrame<PCaloHit> * > cfCaloHits_;
      CrossingFrame<SimTrack> *cfTracks_;
      CrossingFrame<SimVertex> *cfVertices_;

      //      unsigned int eventId_; //=0 for signal, from 1-n for pileup events

      Selector * sel_;
      std::string label_;

    };
}//edm

#endif
