
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimTracker/TrackHistory/interface/HistoryBase.h"


void HistoryBase::traceGenHistory(HepMC::GenParticle const * gpp)
{
    // Third stop criteria: status abs(depth_) particles after the hadronization.
    // The after hadronization is done by detecting the pdg_id pythia code from 88 to 99
    if ( gpp->status() <= abs(depth_) && (gpp->pdg_id() < 88 || gpp->pdg_id() > 99) )
    {
        genParticleTrail_.push_back(gpp);
        // Get the producer vertex.
        HepMC::GenVertex * vertex = gpp->production_vertex();
        // Verify if has a vertex associated
        if ( vertex )
        {
            genVertexTrail_.push_back(vertex);
            if ( vertex->particles_in_size()  ) // Verify if the vertex has incoming particles
                traceGenHistory( *(vertex->particles_in_const_begin()) );
        }
    }
}


bool HistoryBase::traceSimHistory(TrackingParticleRef const & tpr, int depth)
{
    // first stop condition: if the required depth is reached
    if ( depth == depth_ && depth_ >= 0 ) return true;

    // sencond stop condition: if a gen particle is associated to the TP
    if ( !tpr->genParticle().empty() )
    {
        LogDebug("TrackHistory") << "Particle " << tpr->pdgId() << " has a GenParicle image." << std::endl;
        traceGenHistory(&(**(tpr->genParticle_begin())));
        return true;
    }

    LogDebug("TrackHistory") << "No GenParticle image for " << tpr->pdgId() << std::endl;

    // get a reference to the TP's parent vertex and trace it history
    return traceSimHistory( tpr->parentVertex(), depth );
}


bool HistoryBase::traceSimHistory(TrackingVertexRef const & parentVertex, int depth)
{
    // verify if the parent vertex exists
    if ( parentVertex.isNonnull() )
    {
        // save the vertex in the trail
        simVertexTrail_.push_back(parentVertex);

        if ( !parentVertex->sourceTracks().empty() )
        {
            LogDebug("TrackHistory") << "Moving on to the parent particle." << std::endl;

            // select the original source in case of combined vertices
            bool flag = false;
            TrackingVertex::tp_iterator itd, its;

            for (its = parentVertex->sourceTracks_begin(); its != parentVertex->sourceTracks_end(); its++)
            {
                for (itd = parentVertex->daughterTracks_begin(); itd != parentVertex->daughterTracks_end(); itd++)
                    if (itd != its)
                    {
                        flag = true;
                        break;
                    }
                if (flag)
                    break;
            }

            // verify if the new particle is not in the trail (looping partiles)
            if (
                std::find(
                    simParticleTrail_.begin(),
                    simParticleTrail_.end(),
                    *its
                ) != simParticleTrail_.end()
            )
            {
                LogDebug("TrackHistory") <<  "WARNING: Looping track found." << std::endl;
                return false;
            }

            // save particle in the trail
            simParticleTrail_.push_back(*its);
            return traceSimHistory (*its, --depth);
        }
        else
        {
            LogDebug("TrackHistory") <<  "WARNING: Source track for tracking vertex cannot be found." << std::endl;
        }
    }
    else
    {
        LogDebug("TrackHistory") << " WARNING: Vertex cannot be found.";
    }

    return false;   
}

