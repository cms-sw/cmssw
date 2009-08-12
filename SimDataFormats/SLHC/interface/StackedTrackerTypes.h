
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef STACKED_TRACKER_TYPES_H
#define STACKED_TRACKER_TYPES_H

#include "SimDataFormats/SLHC/interface/TrackTriggerHit.h"

#include "SimDataFormats/SLHC/interface/LocalStub.h"
#include "SimDataFormats/SLHC/interface/GlobalStub.h"
#include "SimDataFormats/SLHC/interface/Tracklet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

typedef edm::DetSetVector< TrackTriggerHit > 											TrackTriggerHitCollection;

namespace cmsUpgrades{

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//The reference types
	typedef edm::Ref< edm::PSimHitContainer >											Ref_PSimHit_;
	typedef edm::Ref< edm::DetSetVector<PixelDigi> , PixelDigi >						Ref_PixelDigi_;


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//SimHit type LocalStub
	typedef LocalStub< Ref_PSimHit_ >													LocalStub_PSimHit_;

//SimHit type LocalStub Collections
	typedef std::vector		< LocalStub_PSimHit_ >										LocalStub_PSimHit_Collection;
/*    typedef edm::Ref		< LocalStub_PSimHit_Collection , LocalStub_PSimHit_ >		LocalStub_PSimHit_Ref;
    typedef edm::RefProd	< LocalStub_PSimHit_Collection >							LocalStub_PSimHit_RefProd;
    typedef edm::RefVector	< LocalStub_PSimHit_Collection , LocalStub_PSimHit_ >		LocalStub_PSimHit_RefVector;*/

//SimHit type GlobalStub
	typedef GlobalStub< Ref_PSimHit_ >													GlobalStub_PSimHit_;

//SimHit type GlobalStub Collections
	typedef std::vector		< GlobalStub_PSimHit_ >										GlobalStub_PSimHit_Collection;
/*    typedef edm::Ref		< GlobalStub_PSimHit_Collection , GlobalStub_PSimHit_ >		GlobalStub_PSimHit_Ref;
    typedef edm::RefProd	< GlobalStub_PSimHit_Collection >							GlobalStub_PSimHit_RefProd;
    typedef edm::RefVector	< GlobalStub_PSimHit_Collection , GlobalStub_PSimHit_ >		GlobalStub_PSimHit_RefVector;*/

//SimHit type Tracklet
	typedef Tracklet< Ref_PSimHit_ >													Tracklet_PSimHit_;

//SimHit type Tracklet Collections
	typedef std::vector		< Tracklet_PSimHit_ >										Tracklet_PSimHit_Collection;
/*    typedef edm::Ref		< Tracklet_PSimHit_Collection , Tracklet_PSimHit_ >			Tracklet_PSimHit_Ref;
    typedef edm::RefProd	< Tracklet_PSimHit_Collection >								Tracklet_PSimHit_RefProd;
    typedef edm::RefVector	< Tracklet_PSimHit_Collection , Tracklet_PSimHit_ >			Tracklet_PSimHit_RefVector;*/


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//PixelDigi type LocalStub
	typedef LocalStub< Ref_PixelDigi_ >													LocalStub_PixelDigi_;

//PixelDigi type LocalStub Collections
	typedef std::vector		< LocalStub_PixelDigi_ >									LocalStub_PixelDigi_Collection;
/*    typedef edm::Ref		< LocalStub_PixelDigi_Collection , LocalStub_PixelDigi_ >	LocalStub_PixelDigi_Ref;
    typedef edm::RefProd	< LocalStub_PixelDigi_Collection >							LocalStub_PixelDigi_RefProd;
    typedef edm::RefVector	< LocalStub_PixelDigi_Collection , LocalStub_PixelDigi_ >	LocalStub_PixelDigi_RefVector;*/

//PixelDigi type GlobalStub
	typedef GlobalStub< Ref_PixelDigi_ >												GlobalStub_PixelDigi_;

//PixelDigi type GlobalStub Collections
	typedef std::vector		< GlobalStub_PixelDigi_ >									GlobalStub_PixelDigi_Collection;
/*    typedef edm::Ref		< GlobalStub_PixelDigi_Collection , GlobalStub_PixelDigi_ >	GlobalStub_PixelDigi_Ref;
    typedef edm::RefProd	< GlobalStub_PixelDigi_Collection >							GlobalStub_PixelDigi_RefProd;
    typedef edm::RefVector	< GlobalStub_PixelDigi_Collection , GlobalStub_PixelDigi_ >	GlobalStub_PixelDigi_RefVector;*/

//PixelDigi type Tracklet
	typedef Tracklet< Ref_PixelDigi_ >													Tracklet_PixelDigi_;

//PixelDigi type Tracklet Collections
	typedef std::vector		< Tracklet_PixelDigi_ >										Tracklet_PixelDigi_Collection;
/*    typedef edm::Ref		< Tracklet_PixelDigi_Collection , Tracklet_PixelDigi_ >		Tracklet_PixelDigi_Ref;
    typedef edm::RefProd	< Tracklet_PixelDigi_Collection >							Tracklet_PixelDigi_RefProd;
    typedef edm::RefVector	< Tracklet_PixelDigi_Collection , Tracklet_PixelDigi_ >		Tracklet_PixelDigi_RefVector;*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Finally the TTHit type
typedef edm::Ref< edm::DetSetVector< TrackTriggerHit > , TrackTriggerHit >				Ref_TTHit_;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//TTHit type LocalStub
	typedef LocalStub< Ref_TTHit_ >														LocalStub_TTHit_;

//TTHit type LocalStub Collections
	typedef std::vector		< LocalStub_TTHit_ >										LocalStub_TTHit_Collection;
/*    typedef edm::Ref		< LocalStub_TTHit_Collection , LocalStub_TTHit_ >			LocalStub_TTHit_Ref;
    typedef edm::RefProd	< LocalStub_TTHit_Collection >								LocalStub_TTHit_RefProd;
    typedef edm::RefVector	< LocalStub_TTHit_Collection , LocalStub_TTHit_ >			LocalStub_TTHit_RefVector;*/

//TTHit type GlobalStub
	typedef GlobalStub< Ref_TTHit_ >													GlobalStub_TTHit_;

//TTHit type GlobalStub Collections
	typedef std::vector		< GlobalStub_TTHit_ >										GlobalStub_TTHit_Collection;
/*    typedef edm::Ref		< GlobalStub_TTHit_Collection , GlobalStub_TTHit_ >			GlobalStub_TTHit_Ref;
    typedef edm::RefProd	< GlobalStub_TTHiti_Collection >							GlobalStub_TTHit_RefProd;
    typedef edm::RefVector	< GlobalStub_TTHit_Collection , GlobalStub_TTHit_ >			GlobalStub_TTHit_RefVector;*/

//TTHit type Tracklet
	typedef Tracklet< Ref_TTHit_ >														Tracklet_TTHit_;

//TTHit type Tracklet Collections
	typedef std::vector		< Tracklet_TTHit_ >											Tracklet_TTHit_Collection;
/*    typedef edm::Ref		< Tracklet_TTHit_Collection , Tracklet_TTHit_ >				Tracklet_TTHit_Ref;
    typedef edm::RefProd	< Tracklet_TTHit_Collection >								Tracklet_TTHit_RefProd;
    typedef edm::RefVector	< Tracklet_TTHit_Collection , Tracklet_TTHit_ >				Tracklet_TTHit_RefVector;*/

    // Cluster data types
    typedef std::vector<Ref_PSimHit_> Cluster_PSimHit;
    typedef std::vector<Ref_PixelDigi_> Cluster_PixelDigi;
    typedef std::vector<Ref_TTHit_> Cluster_TTHit;
    
    typedef std::vector<Cluster_PSimHit> Cluster_PSimHit_Collection;
    typedef std::vector<Cluster_PixelDigi> Cluster_PixelDigi_Collection;
    typedef std::vector<Cluster_TTHit> Cluster_TTHit_Collection;

    typedef std::map<std::pair<StackedTrackerDetId,int>,Cluster_PSimHit_Collection> Cluster_PSimHit_Map;
    typedef std::map<std::pair<StackedTrackerDetId,int>,Cluster_PixelDigi_Collection> Cluster_PixelDigi_Map;
    typedef std::map<std::pair<StackedTrackerDetId,int>,Cluster_TTHit_Collection> Cluster_TTHit_Map;
}
#endif
