#ifndef HistoryBase_h
#define HistoryBase_h

#include <set>

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

//! Base class to all the history types.
class HistoryBase
{

public:

    //! HepMC::GenParticle trail type.
    typedef std::vector<const HepMC::GenParticle *> GenParticleTrail;
    
    //! reco::GenParticle trail type.
    typedef std::vector<const reco::GenParticle *> RecoGenParticleTrail;
    
    //!reco::GenParticle trail helper type.
    typedef std::set<const reco::GenParticle *> RecoGenParticleTrailHelper;

    //! GenVertex trail type.
    typedef std::vector<const HepMC::GenVertex *> GenVertexTrail;

    //! GenVertex trail helper type.
    typedef std::set<const HepMC::GenVertex *> GenVertexTrailHelper;

    //! SimParticle trail type.
    typedef std::vector<TrackingParticleRef> SimParticleTrail;

    //! SimVertex trail type.
    typedef std::vector<TrackingVertexRef> SimVertexTrail;

    // Default constructor
    HistoryBase()
    {
        // Default depth
        depth_ = -1;
    }

    //! Set the depth of the history.
    /* Set TrackHistory to given depth. Positive values
       constrain the number of TrackingVertex visit in the history.
       Negatives values set the limit of the iteration over generated
       information i.e. (-1 -> status 1 or -2 -> status 2 particles).

       /param[in] depth the history
    */
    void depth(int d)
    {
        depth_ = d;
    }

    //! Return all the simulated vertices in the history.
    SimVertexTrail const & simVertexTrail() const
    {
        return simVertexTrail_;
    }

    //! Return all the simulated particle in the history.
    SimParticleTrail const & simParticleTrail() const
    {
        return simParticleTrail_;
    }

    //! Return all generated vertex in the history.
    GenVertexTrail const & genVertexTrail() const
    {
        return genVertexTrail_;
    }

    //! Return all generated particle (HepMC::GenParticle) in the history.
    GenParticleTrail const & genParticleTrail() const
    {
        return genParticleTrail_;
    }
    
     //! Return all reco::GenParticle in the history.
    RecoGenParticleTrail const & recoGenParticleTrail() const
    {
        return recoGenParticleTrail_;
    }

    //! Return the initial tracking particle from the history.
    const TrackingParticleRef & simParticle() const
    {
        return simParticleTrail_[0];
    }

    //! Return the initial tracking vertex from the history.
    const TrackingVertexRef & simVertex() const
    {
        return simVertexTrail_[0];
    }

    //! Returns a pointer to most primitive status 1 or 2 particle in the genParticleTrail_.
    const HepMC::GenParticle * genParticle() const
    {
        if ( genParticleTrail_.empty() ) return 0;
        return genParticleTrail_[genParticleTrail_.size()-1];
    }
    
    //! Returns a pointer to most primitive status 1 or 2 particle in the recoGenParticleTrail_.
    const reco::GenParticle * recoGenParticle() const
    {
        if ( recoGenParticleTrail_.empty() ) return 0;
        return recoGenParticleTrail_[recoGenParticleTrail_.size()-1];
    }

protected:

    // History cointainers
    GenVertexTrail genVertexTrail_;
    GenParticleTrail genParticleTrail_;
    RecoGenParticleTrail recoGenParticleTrail_;
    SimVertexTrail simVertexTrail_;
    SimParticleTrail simParticleTrail_;

    // Helper function to speedup search
    GenVertexTrailHelper genVertexTrailHelper_;
    RecoGenParticleTrailHelper recoGenParticleTrailHelper_;

    //! Evaluate track history using a TrackingParticleRef.
    /* Return false when the history cannot be determined upto a given depth.
       If not depth is pass to the function no restriction are apply to it.

       /param[in] TrackingParticleRef of a simulated track
       /param[in] depth of the track history
       /param[out] boolean that is true when history can be determined
    */
    bool evaluate(TrackingParticleRef tpr)
    {
        resetTrails(tpr);
        return traceSimHistory(tpr, depth_);
    }

    //! Evaluate track history using a TrackingParticleRef.
    /* Return false when the history cannot be determined upto a given depth.
       If not depth is pass to the function no restriction are apply to it.

       /param[in] TrackingVertexRef of a simulated vertex
       /param[in] depth of the track history
       /param[out] boolean that is true when history can be determined
    */
    bool evaluate(TrackingVertexRef tvr)
    {
        resetTrails();
        return traceSimHistory(tvr, depth_);
    }

private:

    int depth_;

    //! Trace all the simulated information for a given reference to a TrackingParticle.
    bool traceSimHistory (TrackingParticleRef const &, int);

    //! Trace all the simulated information for a given reference to a TrackingVertex.
    bool traceSimHistory (TrackingVertexRef const &, int);

    //! Trace all the simulated information for a given pointer to a HepMC::GenParticle.
    void traceGenHistory (HepMC::GenParticle const *);
    
    //! Trace all the simulated information for a given pointer to a reco::GenParticle.
    void traceRecoGenHistory (reco::GenParticle const *);
    

    //! Trace all the simulated information for a given pointer to a GenVertex.
    void traceGenHistory (HepMC::GenVertex const *);

    //! Reset trail functions.
    void resetTrails()
    {
        simParticleTrail_.clear();
        simVertexTrail_.clear();
        genVertexTrail_.clear();
        genParticleTrail_.clear();
        recoGenParticleTrail_.clear();
        recoGenParticleTrailHelper_.clear();
        genVertexTrailHelper_.clear();
    }

    void resetTrails(TrackingParticleRef tpr)
    {
        resetTrails();
        simParticleTrail_.push_back(tpr);
    }
};

#endif
