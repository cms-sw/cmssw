#ifdef SLHC_DT_TRK_DFENABLE

#include "SimDataFormats/SLHC/interface/LocalStub.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Now for the implimentation of the helper methods
//Specialize the template for PSimHits
	template<>
	GlobalPoint cmsUpgrades::LocalStub< edm::Ref<edm::PSimHitContainer> >::hitPosition(const GeomDetUnit* geom, const edm::Ref<edm::PSimHitContainer> &hit) const
	{
		return geom->surface().toGlobal( hit->localPosition() ) ;
	}

//Default assumes pixelization
//	template<	typename T	>
//	GlobalPoint cmsUpgrades::LocalStub<T>::hitPosition(const GeomDetUnit* geom, const T &hit) const
//	{
//		MeasurementPoint mp( hit->row() + 0.5, hit->column() + 0.5 ); // Add 0.5 to get the center of the pixel.
//		return geom->surface().toGlobal( geom->topology().localPosition( mp ) ) ;
//	}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
