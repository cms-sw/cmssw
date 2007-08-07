/* \class PtMinElectronCountFilter
 *
 * Filters events if at least N electrons above 
 * a pt cut are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<
          reco::PixelMatchGsfElectronCollection, PtMinSelector
        > PtMinPixelMatchGsfElectronCountFilter;

DEFINE_FWK_MODULE( PtMinPixelMatchGsfElectronCountFilter );
