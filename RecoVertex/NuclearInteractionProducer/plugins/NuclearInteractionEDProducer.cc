#include "RecoVertex/NuclearInteractionProducer/interface/NuclearInteractionEDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/VertexReco/interface/NuclearInteractionFwd.h"

#include "RecoVertex/NuclearInteractionProducer/interface/NuclearVertexBuilder.h"
#include "RecoVertex/NuclearInteractionProducer/interface/NuclearLikelihood.h"


#include "FWCore/Framework/interface/EventSetup.h"

namespace {
  void print(std::ostringstream& str, const reco::NuclearInteraction& nucl, const NuclearVertexBuilder& builder);
}


NuclearInteractionEDProducer::NuclearInteractionEDProducer(const edm::ParameterSet& iConfig) : 
conf_(iConfig)
{
  token_primaryTrack = consumes<reco::TrackCollection>(iConfig.getParameter<std::string>("primaryProducer"));
  token_secondaryTrack = consumes<reco::TrackCollection>(iConfig.getParameter<std::string>("secondaryProducer"));
  token_additionalSecTracks = consumes<reco::TrackCollection>(iConfig.getParameter<std::string>("additionalSecondaryProducer"));
  token_primaryTrajectory = consumes<TrajectoryCollection>(iConfig.getParameter<std::string>("primaryProducer"));
  token_refMapH = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<std::string>("primaryProducer"));
  token_nuclMapH = consumes<TrajectoryToSeedsMap>(iConfig.getParameter<std::string>("seedsProducer"));

  produces<reco::NuclearInteractionCollection>();
}

NuclearInteractionEDProducer::~NuclearInteractionEDProducer()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
NuclearInteractionEDProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if ( magFieldWatcher_.check(iSetup) || transientTrackWatcher_.check(iSetup) ) {
    /// Get magnetic field
    edm::ESHandle<MagneticField> magField;
    iSetup.get<IdealMagneticFieldRecord>().get(magField);

    edm::ESHandle<TransientTrackBuilder> builder;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",builder);

    vertexBuilder = std::make_unique< NuclearVertexBuilder >(magField.product(), builder.product(), conf_);
    likelihoodCalculator = std::make_unique< NuclearLikelihood >();
  }

   /// Get the primary tracks
   edm::Handle<reco::TrackCollection>  primaryTrackCollection;
   iEvent.getByToken(token_primaryTrack, primaryTrackCollection);

   /// Get the primary trajectories (produced by the Refitter)
   edm::Handle< TrajectoryCollection > primaryTrajectoryCollection;
   iEvent.getByToken(token_primaryTrajectory, primaryTrajectoryCollection);

   /// Get the AssociationMap between primary tracks and trajectories
   edm::Handle< TrajTrackAssociationCollection > refMapH;
   iEvent.getByToken(token_refMapH, refMapH);
   const TrajTrackAssociationCollection& refMap = *(refMapH.product());

   /// Get the AssociationMap between seeds and primary trajectories
   edm::Handle<TrajectoryToSeedsMap>  nuclMapH;
   iEvent.getByToken(token_nuclMapH, nuclMapH);
   const TrajectoryToSeedsMap& nuclMap = *(nuclMapH.product());

   /// Get the secondary tracks
   edm::Handle<reco::TrackCollection>  secondaryTrackCollection;
   iEvent.getByToken(token_secondaryTrack, secondaryTrackCollection);

   // Get eventual additional secondary tracks
   edm::Handle<reco::TrackCollection>  additionalSecTracks;
   iEvent.getByToken(token_additionalSecTracks, additionalSecTracks);

   /// Definition of the output
   std::auto_ptr<reco::NuclearInteractionCollection> theNuclearInteractions(new reco::NuclearInteractionCollection);

   typedef edm::Ref<TrajectoryCollection> TrajectoryRef;

   /// Loop on all primary trajectories
   for(unsigned int i = 0; i < primaryTrajectoryCollection->size() ; i++) {

         TrajectoryRef  trajRef( primaryTrajectoryCollection, i );

         /// 1. Get the primary track from the trajectory
         TrajTrackAssociationCollection::const_iterator itPrimTrack = refMap.find( trajRef );
         if( itPrimTrack == refMap.end() || (itPrimTrack->val).isNull() ) continue;
         const reco::TrackRef& primary_track = itPrimTrack->val;

         /// 2. Get the seeds from the trajectory
         TrajectoryToSeedsMap::const_iterator itSeeds = nuclMap.find( trajRef );
         if( itSeeds == nuclMap.end() || (itSeeds->val).isNull()) continue; 
         const TrajectorySeedRefVector& seeds = itSeeds->val;

         /// 3. Get the secondary tracks
         std::vector<reco::TrackRef> secondary_tracks;
         for( unsigned int k=0; k < secondaryTrackCollection->size(); k++) {
                     reco::TrackRef currentTrk(secondaryTrackCollection, k);
                     if( isInside( currentTrk, seeds ) ) secondary_tracks.push_back(currentTrk);
         }
              
         /// 4. Get the vertex and the likelihood
         vertexBuilder->build(primary_track, secondary_tracks);
         likelihoodCalculator->calculate( vertexBuilder->getVertex() );

         /// 5. Build the nuclear interaction
         reco::NuclearInteraction nuclInter(seeds, vertexBuilder->getVertex(), likelihoodCalculator->result() );

         /// 6. Search for additional secondary tracks in an other track collection
         ///    and recompute the nuclear interaction
         if( additionalSecTracks.isValid() )
                 findAdditionalSecondaryTracks(nuclInter, additionalSecTracks); 

         theNuclearInteractions->push_back( nuclInter );

         std::ostringstream  str;
         print(str, nuclInter, *vertexBuilder);
         edm::LogInfo("NuclearInteractionMaker") << str.str();

   }


   LogDebug("NuclearInteractionMaker") << "End of NuclearInteractionMaker - Number of nuclear interactions found :" << theNuclearInteractions->size();
   iEvent.put(theNuclearInteractions);
}

// ------ method used to check whether the seed of a track belong to the vector of seeds --
bool NuclearInteractionEDProducer::isInside( const reco::TrackRef& track, const TrajectorySeedRefVector& seeds) {
    unsigned int seedKey = track->seedRef().key();
    for (unsigned int i=0; i< seeds.size(); i++) { if( seeds[i].key() == seedKey ) return true; }
    return false;
}

void NuclearInteractionEDProducer::findAdditionalSecondaryTracks( reco::NuclearInteraction& nucl, 
                                      const edm::Handle<reco::TrackCollection>& additionalSecTracks) const {
      
      LogDebug("NuclearInteractionMaker") << "Check if one of the " << additionalSecTracks->size() 
                                          << " additional secondary track is compatible";
      reco::TrackRefVector allSecondary;
      for(unsigned int i=0; i< additionalSecTracks->size(); i++) {
                 reco::TrackRef sec(additionalSecTracks, i);
                 if( vertexBuilder->isCompatible( sec ) ) {
                      // check if sec is already a secondary track (with id = tkId) 
                      // else add this new track
                      vertexBuilder->addSecondaryTrack( sec );
                 }
      }
      likelihoodCalculator->calculate( vertexBuilder->getVertex() );

      nucl = reco::NuclearInteraction(nucl.seeds(), vertexBuilder->getVertex(), likelihoodCalculator->result() );
}

// -- print out
namespace {
void print(std::ostringstream& out, const reco::NuclearInteraction& nucl, const NuclearVertexBuilder& builder) {
   out<<"Nuclear Interaction with vertex position : (";
   out<< nucl.vertex().position().x() << " , "
      << nucl.vertex().position().y() << " , "
      << nucl.vertex().position().z() << ")";
   reco::TrackRef primTrack = (nucl.primaryTrack()).castTo<reco::TrackRef>();
   out<<"\tLikelihood : " << nucl.likelihood() << std::endl;
   out<<"\tPrimary Track : Pt = " << primTrack->pt() << "  - Nhits = "
      << primTrack->numberOfValidHits() << std::endl;
   out << "\tNumber of seeds : " << nucl.seedsSize() << std::endl;
   out << "\tNumber of secondary Tracks : " << nucl.secondaryTracksSize() << std::endl;
   int it=0;
   for( reco::NuclearInteraction::trackRef_iterator itr_=nucl.secondaryTracks_begin(); itr_ != nucl.secondaryTracks_end(); itr_++, it++) {
                reco::TrackRef secTrack = (*itr_).castTo<reco::TrackRef>();
                ClosestApproachInRPhi* theApproach = builder.closestApproach(primTrack, secTrack);
                out << "\t\t Secondary track " << it << " : Pt = " << (*itr_)->pt() 
                    << " - Nhits = " << (*itr_)->numberOfValidHits()
                    << " - Dist = " << theApproach->distance()
                    << " - chi2 = " << (*itr_)->normalizedChi2() << std::endl;
                delete theApproach;
      }
   out << "----------------" << std::endl;
}
}
