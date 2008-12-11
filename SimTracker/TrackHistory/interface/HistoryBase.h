#ifndef HistoryBase_h
#define HistoryBase_h

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

//! Base class to all the history types.
class HistoryBase
{

public:

    //! GenParticle trail type.
    typedef std::vector<const HepMC::GenParticle *> GenParticleTrail;

    //! GenVertex trail type.
    typedef std::vector<const HepMC::GenVertex *> GenVertexTrail;

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

    //! Return all generated particle in the history.
    GenParticleTrail const & genParticleTrail() const
    {
        return genParticleTrail_;
    }
   
protected:

    // History cointainers
    GenVertexTrail genVertexTrail_;
    GenParticleTrail genParticleTrail_;
    SimVertexTrail simVertexTrail_;
    SimParticleTrail simParticleTrail_;

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
        resetTrails(tvr);
        return traceSimHistory(tvr, depth_);
    }
        
private:

    int depth_;

    //! Trace all the simulated information for a given reference to a TrackingParticle.
    bool traceSimHistory (TrackingParticleRef const &, int);

    //! Trace all the simulated information for a given reference to a TrackingVertex.
    bool traceSimHistory (TrackingVertexRef const &, int);

    //! Trace all the simulated information for a given pointer to a GenParticle.
    void traceGenHistory (HepMC::GenParticle const *);

    //! Trace all the simulated information for a given pointer to a GenVertex.
    void traceGenHistory (HepMC::GenVertex const *);

    //! Reset trail functions.
    void resetTrails()
    {
        simParticleTrail_.clear();
        simVertexTrail_.clear();
        genVertexTrail_.clear();
        genParticleTrail_.clear();
    }            

    void resetTrails(TrackingParticleRef tpr)
    {
    	resetTrails();
    	simParticleTrail_.push_back(tpr);
    }
    
    void resetTrails(TrackingVertexRef tvr)
    {
    	resetTrails();
    	simVertexTrail_.push_back(tvr);
    }    
};

#endif
