#ifndef MixingModule_h
#define MixingModule_h

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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Selector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

#include "MixingWorkerBase.h"

#include <vector>
#include <string>

class CrossingFramePlaybackInfo;
class MixingWorkerBase;
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
 
    private:
      virtual bool verifyRegistry(std::string object, std::string subdet,InputTag &tag,std::string &label);
      virtual void put(edm::Event &e) ;
      virtual void createnewEDProduct();
      virtual void addSignals(const edm::Event &e); 
      virtual void doPileUp(edm::Event &e);
      virtual void addPileups(const int bcr, edm::Event*,unsigned int EventId,unsigned int worker);
      virtual void setEventStartInfo(const unsigned int s); // set in CF-s
      virtual void getEventStartInfo(edm::Event & e, const unsigned int s); // fill in in base class

      std::string labelPlayback_;
      CrossingFramePlaybackInfo *playbackInfo_;

      Selector * sel_;

      std::vector<MixingWorkerBase *> workers_;
      std::vector<std::string> wantedBranches_;

    };
}//edm

#endif
