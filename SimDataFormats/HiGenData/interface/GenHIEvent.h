#ifndef SimDataFormats_GenHIEvent_h
#define SimDataFormats_GenHIEvent_h

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

namespace edm {
   class GenHIEvent {
   public:

      typedef std::vector<std::vector<reco::GenParticleRef> > SubEventCollection;

      GenHIEvent() :
	 b_(-99),
	 npart_(-99),
	 ncoll_(-99),
	 nhard_(-99),
	 phi_(-99),
	 nCharged_(-99),
	 nChargedMR_(-99),
	 meanPt_(-99),
	 meanPtMR_(-99)
	    {
	       subevents_.reserve(0);
	       ;}

	 GenHIEvent(double b, int npart, int ncoll, int nhard, double phi,double nCharged,double nChargedMR,double meanPt,double meanPtMR) : 
	 b_(b), 
	 npart_(npart), 
	 ncoll_(ncoll), 
	 nhard_(nhard), 
	 phi_(phi),
	 nCharged_(nCharged),
         nChargedMR_(nChargedMR),
         meanPt_(meanPt),
         meanPtMR_(meanPtMR)
	    {
               subevents_.reserve(0);
	       ;}




      virtual                    ~GenHIEvent() {}

      double b() const {return b_;}
      int Npart() const {return npart_;}
      int Ncoll() const {return ncoll_;}
      int Nhard() const {return nhard_;}
      double evtPlane() const {return phi_;}
      int Ncharged() const {return nCharged_;}
      int NchargedMR() const {return nChargedMR_;}
      double MeanPt() const {return meanPt_;}
      double MeanPtMR() const {return meanPtMR_;}

      void setGenParticles(const reco::GenParticleCollection*) const;
      const std::vector<reco::GenParticleRef>    getSubEvent(unsigned int sub_id) const;

      int           getNsubs() const {return subevents_.size();}

   private:

      mutable SubEventCollection subevents_;
      int sel_;

      double b_;
      int npart_;
      int ncoll_;
      int nhard_;
      double phi_;

      int nCharged_;
      int nChargedMR_;
      double meanPt_;
      double meanPtMR_;



   };
}
#endif
