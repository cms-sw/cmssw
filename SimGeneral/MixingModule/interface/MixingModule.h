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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Mixing/Base/interface/BMixingModule.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
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

    private:

      virtual void put(edm::Event &e) ;
      virtual void createnewEDProduct();
      virtual void addSignals(edm::Event &e); 
      virtual void addPileups(const int bcr, edm::Event*);

      // internally used information
      std::vector<std::string> trackerSubdetectors_;
      std::vector<std::string> caloSubdetectors_;
      CrossingFrame *simcf_;

    };
}//edm

#endif
