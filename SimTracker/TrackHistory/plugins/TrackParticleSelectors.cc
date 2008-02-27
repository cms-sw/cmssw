#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

#include "SimTracker/TrackHistory/plugins/TrackParticleSelector.h"

namespace reco{
  namespace modules {

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackingParticleCollection, TrackCategories::PV> > PVTrackingParticleSelection;    
    // declare the module as plugin
    DEFINE_FWK_MODULE( PVTrackingParticleSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackingParticleCollection, TrackCategories::SV> > SVTrackingParticleSelection;    
    // declare the module as plugin
    DEFINE_FWK_MODULE( SVTrackingParticleSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackingParticleCollection, TrackCategories::TV> > TVTrackingParticleSelection;    
    // declare the module as plugin
    DEFINE_FWK_MODULE( TVTrackingParticleSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackingParticleCollection, TrackCategories::Light> > LightTrackingParticleSelection;    
    // declare the module as plugin
    DEFINE_FWK_MODULE( LightTrackingParticleSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackingParticleCollection, TrackCategories::Charm> > CTrackingParticleSelection;    
    // declare the module as plugin
    DEFINE_FWK_MODULE( CTrackingParticleSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackingParticleCollection, TrackCategories::Bottom> > BTrackingParticleSelection;    
    // declare the module as plugin
    DEFINE_FWK_MODULE( BTrackingParticleSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackCollection, TrackCategories::PV> > PVTrackSelection;    
    // declare the module as plugin
    DEFINE_FWK_MODULE( PVTrackSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackCollection, TrackCategories::SV> > SVTrackSelection;
    // declare the module as plugin
    DEFINE_FWK_MODULE( SVTrackSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackCollection, TrackCategories::TV> > TVTrackSelection;
    // declare the module as plugin
    DEFINE_FWK_MODULE( TVTrackSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackCollection, TrackCategories::Light> > LightTrackSelection;    
    // declare the module as plugin
    DEFINE_FWK_MODULE( LightTrackSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackCollection, TrackCategories::Charm> > CTrackSelection;
    // declare the module as plugin
    DEFINE_FWK_MODULE( CTrackSelection );

    // define your producer name
    typedef ObjectSelector<TrackParticleSelector<TrackCollection, TrackCategories::Bottom> > BTrackSelection;
    // declare the module as plugin
    DEFINE_FWK_MODULE( BTrackSelection );
  }
}
