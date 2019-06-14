/*! \brief   Definition of all the relevant data types
 *  \details Herw we declare instances of all the relevant types. 
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

namespace {
  namespace {
    // edm::Ptr< TrackingParticle >                  TP;
    // std::vector< edm::Ptr< TrackingParticle > > V_TP;

    TTClusterAssociationMap<Ref_Phase2TrackerDigi_> CAM_PD;
    edm::Wrapper<TTClusterAssociationMap<Ref_Phase2TrackerDigi_> > W_CAM_PD;

    std::map<edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_> >, TTCluster<Ref_Phase2TrackerDigi_> >,
             std::vector<edm::Ptr<TrackingParticle> > >
        M_CAM_C_TP_PD;
    std::map<edm::Ptr<TrackingParticle>,
             std::vector<edm::Ref<edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_> >,
                                  TTCluster<Ref_Phase2TrackerDigi_> > > >
        M_CAM_TP_C_PD;
    edm::RefProd<TTClusterAssociationMap<Ref_Phase2TrackerDigi_> > R_CAM_PD;

    TTStubAssociationMap<Ref_Phase2TrackerDigi_> SAM_PD;
    edm::Wrapper<TTStubAssociationMap<Ref_Phase2TrackerDigi_> > W_SAM_PD;

    std::map<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> >,
             edm::Ptr<TrackingParticle> >
        M_SAM_S_TP_PD;
    std::map<
        edm::Ptr<TrackingParticle>,
        std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > > >
        M_SAM_TP_S_PD;
    edm::RefProd<TTStubAssociationMap<Ref_Phase2TrackerDigi_> > R_SAM_PD;

    TTTrackAssociationMap<Ref_Phase2TrackerDigi_> TAM_PD;
    edm::Wrapper<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > W_TAM_PD;
    edm::RefProd<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > R_TAM_PD;

    /*
    std::pair< edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > >, edm::Ptr< TrackingParticle > >                P_TAM_S_TP_PD;
    std::pair< edm::Ptr< TrackingParticle >, std::vector< edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > > > > P_TAM_TP_S_PD;

    std::map< edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > >, edm::Ptr< TrackingParticle > >                M_TAM_S_TP_PD;
    std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ptr< TTTrack< Ref_Phase2TrackerDigi_ > > > > M_TAM_TP_S_PD;
    */
  }  // namespace
}  // namespace
