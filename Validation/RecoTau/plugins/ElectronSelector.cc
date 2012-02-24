 #include "FWCore/Framework/interface/MakerMacros.h"
 #include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
 #include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
 #include "DataFormats/Common/interface/View.h"
 #include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
 #include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
 
  typedef SingleObjectSelector<
			edm::View<reco::GsfElectron>, 
			StringCutObjectSelector<reco::GsfElectron>,
			reco::GsfElectronCollection
		  > ElectronSelector;
 
 DEFINE_FWK_MODULE( ElectronSelector );

