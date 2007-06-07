
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"

#include <vector>


typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopElectron>,
            SingleObjectSelector<TopElectron>
          >
        > TopElectronSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopMuon>,
            SingleObjectSelector<TopMuon>
          >
        > TopMuonSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopJet>,
            SingleObjectSelector<TopJet>
          >
        > TopJetSelector;
typedef ObjectSelector<
          SingleElementCollectionSelector<
            std::vector<TopMET>,
            SingleObjectSelector<TopMET>
          >
        > TopMETSelector;
