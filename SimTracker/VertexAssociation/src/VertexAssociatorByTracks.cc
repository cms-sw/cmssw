#include "SimTracker/VertexAssociation/interface/VertexAssociatorByTracks.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

/* Constructor */
VertexAssociatorByTracks::VertexAssociatorByTracks(const edm::EDProductGetter *productGetter,
                                                   double R2SMatchedSimRatio,
                                                   double R2SMatchedRecoRatio,
                                                   double S2RMatchedSimRatio,
                                                   double S2RMatchedRecoRatio,
                                                   const TrackingParticleSelector *selector,
                                                   reco::TrackBase::TrackQuality trackQuality,
                                                   const reco::RecoToSimCollection *trackRecoToSimAssociation,
                                                   const reco::SimToRecoCollection *trackSimToRecoAssociation):
  productGetter_(productGetter),
  R2SMatchedSimRatio_(R2SMatchedSimRatio),
  R2SMatchedRecoRatio_(R2SMatchedRecoRatio),
  S2RMatchedSimRatio_(S2RMatchedSimRatio),
  S2RMatchedRecoRatio_(S2RMatchedRecoRatio),
  selector_(selector),
  trackQuality_(trackQuality),
  trackRecoToSimAssociation_(trackRecoToSimAssociation),
  trackSimToRecoAssociation_(trackSimToRecoAssociation)
{}

/* Destructor */
VertexAssociatorByTracks::~VertexAssociatorByTracks() {}

reco::VertexRecoToSimCollection VertexAssociatorByTracks::associateRecoToSim(
    const edm::Handle<edm::View<reco::Vertex> > & recoVertexes,
    const edm::Handle<TrackingVertexCollection> & trackingVertexes
) const
{
    reco::VertexRecoToSimCollection  outputCollection(productGetter_);

    std::map<TrackingVertexRef,std::pair<double, std::size_t> > matches;

    LogDebug("VertexAssociation") << "reco::VertexCollection size = " << recoVertexes->size()
                                  << " ; TrackingVertexCollection size = " << trackingVertexes->size() << std::endl;

    // Loop over RecoVertex
    for (std::size_t recoIndex = 0; recoIndex < recoVertexes->size(); ++recoIndex)
    {
        reco::VertexBaseRef recoVertex(recoVertexes, recoIndex); 

        matches.clear();

        double recoDaughterWeight = 0.;

        // Loop over daughter tracks of RecoVertex
        for (
            reco::Vertex::trackRef_iterator recoDaughter = recoVertex->tracks_begin();
            recoDaughter != recoVertex->tracks_end();
            ++recoDaughter
        )
        {
            // Check the quality of the RecoDoughters
            if ( !(*recoDaughter)->quality(trackQuality_) ) continue;

            // Check for association for the given RecoDaughter
            if ( trackRecoToSimAssociation_->numberOfAssociations(*recoDaughter) > 0 )
            {
                std::vector<std::pair<TrackingParticleRef,double> > associations = (*trackRecoToSimAssociation_)[*recoDaughter];

                // Loop over TrackingParticles associated with RecoDaughter
                for (
                    std::vector<std::pair<TrackingParticleRef,double> >::const_iterator association = associations.begin();
                    association != associations.end();
                    ++association
                )
                {
                    // Get a reference to parent vertex of TrackingParticle associated to the RecoDaughter
                    TrackingVertexRef trackingVertex = association->first->parentVertex();
                    // Store matched RecoDaughter to the trackingVertex
                    matches[trackingVertex].first += recoVertex->trackWeight(*recoDaughter);
                    matches[trackingVertex].second++;
                }
            }

            recoDaughterWeight += recoVertex->trackWeight(*recoDaughter);
        } //loop over daughter tracks of RecoVertex

        std::size_t assoIndex = 0;

        // Loop over map between TrackingVertexes and matched RecoDaugther
        for (
            std::map<TrackingVertexRef,std::pair<double, std::size_t> >::const_iterator match = matches.begin();
            match != matches.end();
            ++match
        )
        {
            // Getting the TrackingVertex information
            TrackingVertexRef trackingVertex = match->first;
            double matchedDaughterWeight = match->second.first;
            std::size_t matchedDaughterCounter = match->second.second;

            // Count for only reconstructible SimDaughters           
            std::size_t simDaughterCounter = 0;

            for (
                TrackingVertex::tp_iterator simDaughter = trackingVertex->daughterTracks_begin();
                simDaughter != trackingVertex->daughterTracks_end();
                ++simDaughter
            )
                if ( (*selector_)(**simDaughter) ) simDaughterCounter++;

            // Sanity condition in case that reconstructable condition is too tight
            if ( simDaughterCounter < matchedDaughterCounter )
                simDaughterCounter = matchedDaughterCounter;

            // Condition over S2RMatchedSimRatio
            if ( (double)matchedDaughterCounter/simDaughterCounter < R2SMatchedSimRatio_ ) continue;

            double quality = (double)matchedDaughterWeight/recoDaughterWeight;

            // Condition over R2SMatchedRecoRatio
            if (quality < R2SMatchedRecoRatio_) continue;

            outputCollection.insert(recoVertex, std::make_pair(trackingVertex, quality));

            LogTrace("VertexAssociation") << "R2S: INDEX " << assoIndex
                                          << " ; SimDaughterCounter = " << simDaughterCounter
                                          << " ; RecoDaughterWeight = " << recoDaughterWeight
                                          << " ; MatchedDaughterCounter = " << matchedDaughterCounter
                                          << " ; MatchedDaughterWeight = " << matchedDaughterWeight
                                          << " ; quality = " << quality;

            assoIndex++;
        }
    } // Loop on RecoVertex

    LogTrace("VertexAssociation") << "\nRecoToSim OUTPUT COLLECTION: outputCollection.size() = " << outputCollection.size() << std::endl;

    return outputCollection;
}



reco::VertexSimToRecoCollection VertexAssociatorByTracks::associateSimToReco(
    const edm::Handle<edm::View<reco::Vertex> > & recoVertexes,
    const edm::Handle<TrackingVertexCollection> & trackingVertexes
) const
{
    reco::VertexSimToRecoCollection  outputCollection(productGetter_);

    // Loop over TrackingVertexes
    std::map<std::size_t,std::pair<double, std::size_t> > matches;

    // Loop over TrackingVertexes
    for (std::size_t simIndex = 0; simIndex < trackingVertexes->size(); ++simIndex)
    {
        TrackingVertexRef trackingVertex(trackingVertexes, simIndex);

        matches.clear();

        std::size_t simDaughterCounter = 0;

        // Loop over daughter tracks of TrackingVertex
        for (
            TrackingVertex::tp_iterator simDaughter = trackingVertex->daughterTracks_begin();
            simDaughter != trackingVertex->daughterTracks_end();
            ++simDaughter
        )
        {
            // Select only reconstructible SimDaughters
            if ( !(*selector_)(**simDaughter) ) continue;

            // Check for association for the given RecoDaughter
            if ( trackSimToRecoAssociation_->numberOfAssociations(*simDaughter) > 0 )
            {
                std::vector<std::pair<reco::TrackBaseRef, double> > associations = (*trackSimToRecoAssociation_)[*simDaughter];

                // Loop over RecoTracks associated with TrackingParticle
                for (
                    std::vector<std::pair<reco::TrackBaseRef,double> >::const_iterator association = associations.begin();
                    association != associations.end();
                    ++association
                )
                {
                    reco::TrackBaseRef recoTrack = association->first;

                    for (std::size_t recoIndex = 0; recoIndex < recoVertexes->size(); ++recoIndex)
                    {
                        reco::VertexBaseRef recoVertex(recoVertexes, recoIndex);

                        for (
                            reco::Vertex::trackRef_iterator recoDaughter = recoVertex->tracks_begin();
                            recoDaughter != recoVertex->tracks_end();
                            ++recoDaughter
                        )
                        {
                            // Store matched RecoDaughter to the RecoVertex
                            if (
                                recoDaughter->id() == recoTrack.id() &&
                                recoDaughter->key() == recoTrack.key()
                            )
                            {
                                matches[recoIndex].first += recoVertex->trackWeight(*recoDaughter);
                                matches[recoIndex].second++; 
                            }
                        }
                    }
                } // loop a recotracks
            }

            simDaughterCounter++;
        } // loop a simDaughter

        std::size_t assoIndex = 0;

        // Loop over map, set score, add to outputCollection
        for (
            std::map<std::size_t,std::pair<double,std::size_t> >::const_iterator match = matches.begin();
            match != matches.end();
            ++match
        )
        {
            // Getting the TrackingVertex information
            reco::VertexBaseRef recoVertex(recoVertexes, match->first);
            double matchedDaughterWeight = match->second.first;
            std::size_t matchedDaughterCounter = match->second.second;

            double recoDaughterWeight = 0.;

            // Weighted count of those tracks with a given quality of the RecoDoughters
            for (
                reco::Vertex::trackRef_iterator recoDaughter = recoVertex->tracks_begin();
                recoDaughter != recoVertex->tracks_end();
                ++recoDaughter
            )
                if ( (*recoDaughter)->quality(trackQuality_) ) 
                    recoDaughterWeight += recoVertex->trackWeight(*recoDaughter);

            // Sanity condition in case that track quality condition is too tight
            if ( recoDaughterWeight < matchedDaughterWeight )
                recoDaughterWeight = matchedDaughterWeight;

            // Condition over S2RMatchedRecoRatio
            if ( matchedDaughterWeight/recoDaughterWeight < S2RMatchedRecoRatio_ ) continue;

            double quality = (double)matchedDaughterCounter/simDaughterCounter;

            // Condition over S2RMatchedSimRatio
            if (quality < S2RMatchedSimRatio_) continue;

            outputCollection.insert(trackingVertex, std::make_pair(recoVertex, quality));

            LogTrace("VertexAssociation") << "R2S: INDEX " << assoIndex
                                          << " ; SimDaughterCounter = " << simDaughterCounter
                                          << " ; RecoDaughterWeight = " << recoDaughterWeight
                                          << " ; MatchedDaughterCounter = " << matchedDaughterCounter
                                          << " ; MatchedDaughterWeight = " << matchedDaughterWeight
                                          << " ; quality = " << quality;

            assoIndex++;
        }
    } // loop over TrackingVertexes

    LogTrace("VertexAssociation") << "SimToReco OUTPUT COLLECTION: outputCollection.size() = " << outputCollection.size() << std::endl;

    return outputCollection;
}

