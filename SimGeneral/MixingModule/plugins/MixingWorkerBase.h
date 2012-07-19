#ifndef MixingWorkerBase_h
#define MixingWorkerBase_h

/** \class MixingWorkerBase
 *
 * MixingWorkerBase is an auxiliary class for the MixingModule
 *
 * \author Ursula Berthon, LLR Palaiseau
 *
 * \version   1st Version JMarch 2008

 *
 ************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Principal.h"
#include "Mixing/Base/interface/PileUp.h"
#include "DataFormats/Provenance/interface/EventID.h"

namespace edm
{
  class MixingModule;
  class EventSetup;

  /*! This class allows MixingModule to store a vector of
   *  MixingWorkers, which are templated.
   */
  class MixingWorkerBase 
    {
    public:

      /*Normal constructor*/ 
      MixingWorkerBase() {}

      /**Default destructor*/
      virtual ~MixingWorkerBase();

      /**Steps in BMixingModule::produce*/
      virtual bool checkSignal(const edm::Event &e)=0;
      virtual void createnewEDProduct()=0; 
      virtual void addSignals(const edm::Event &e) =0;
      virtual void addPileups(const int bcr, const edm::EventPrincipal&,
			      unsigned int EventNr, int vertexOffset=0)=0;
      virtual void setBcrOffset()=0;
      virtual void setSourceOffset(const unsigned int s)=0;
      virtual void setTof()=0;
      virtual void put(edm::Event &e) =0;
      virtual void reload(const edm::EventSetup & setup){};
    };
}//edm

#endif
