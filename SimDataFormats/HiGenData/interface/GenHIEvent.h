#ifndef SimDataFormats_GenHIEvent_h
#define SimDataFormats_GenHIEvent_h

#include "SimDataFormats/HiGenData/interface/SubEvent.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenEvent.h"

namespace edm {
   
   class GenHIEvent {
   public:
      GenHIEvent() : evts_(0), sel_(0) {}
      GenHIEvent(std::vector<SubEvent> evs , int x = 0) : evts_(evs), sel_(x) {}
      virtual                    ~GenHIEvent() {}
      SubEvent                  getSubEvent(int sub_id) const;
      std::vector<HepMC::GenParticle*>    getParticles(const HepMC::GenEvent& evt, int sub_id) const;
      const HepMC::GenVertex*                 getVertex(const HepMC::GenEvent& evt, int sub_id) const;
      int           getNsubs() const {return evts_.size();}

   private:
      std::vector<SubEvent> evts_;
      int sel_;
   };
}
#endif
