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

namespace
{
  namespace
  {
    edm::Ptr< TrackingParticle >                  TP;
    std::vector< edm::Ptr< TrackingParticle > > V_TP;

    TTClusterAssociationMap< Ref_PixelDigi_ >                   CAM_PD;
    edm::Wrapper< TTClusterAssociationMap< Ref_PixelDigi_ > > W_CAM_PD;

    std::map< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > >, std::vector< edm::Ptr< TrackingParticle > > > M_CAM_C_TP_PD;
    std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > > > M_CAM_TP_C_PD;
    edm::RefProd< TTClusterAssociationMap< Ref_PixelDigi_ > >                                                                                                  R_CAM_PD;

    TTStubAssociationMap< Ref_PixelDigi_ >                   SAM_PD;
    edm::Wrapper< TTStubAssociationMap< Ref_PixelDigi_ > > W_SAM_PD;

    std::map< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > >, edm::Ptr< TrackingParticle > >                M_SAM_S_TP_PD;
    std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > > M_SAM_TP_S_PD;
    edm::RefProd< TTStubAssociationMap< Ref_PixelDigi_ > >                                                                                               R_SAM_PD;

    TTTrackAssociationMap< Ref_PixelDigi_ >                   TAM_PD;
    edm::Wrapper< TTTrackAssociationMap< Ref_PixelDigi_ > > W_TAM_PD;

    std::map< edm::Ptr< TTTrack< Ref_PixelDigi_ > >, edm::Ptr< TrackingParticle > >                M_TAM_S_TP_PD;
    std::map< edm::Ptr< TrackingParticle >, std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > > M_TAM_TP_S_PD;
  }
}

