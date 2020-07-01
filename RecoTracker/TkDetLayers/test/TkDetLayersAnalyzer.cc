// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"

// ======= specific includes =======
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"

// for trie
#include "Geometry/TrackerGeometryBuilder/interface/trackerHierarchy.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/Common/interface/Trie.h"

// for the test
#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"

namespace {

  // FIXME here just to allow prototyping...
  namespace trackerTrie {
    typedef GeomDet const* PDet;
    typedef edm::Trie<PDet> DetTrie;
    typedef edm::TrieNode<PDet> Node;
    typedef Node const* node_pointer;  // sigh....
    typedef edm::TrieNodeIter<PDet> node_iterator;
  }  // namespace trackerTrie

  // Wrapper for trie call back
  template <typename F>
  struct WrapTrieCB {
    WrapTrieCB(F& fi) : f(fi) {}
    template <typename P>
    void operator()(P p, std::string const&) {
      f(*p);
    }

    F& f;
  };

}  // namespace

using namespace edm;
using namespace std;

class TkDetLayersAnalyzer : public edm::one::EDAnalyzer<> {
public:
  TkDetLayersAnalyzer(const edm::ParameterSet&);
  ~TkDetLayersAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

TkDetLayersAnalyzer::TkDetLayersAnalyzer(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
}

TkDetLayersAnalyzer::~TkDetLayersAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void TkDetLayersAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ESHandle<TrackerGeometry> pTrackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(pTrackerGeometry);

  ESHandle<GeometricDet> pDD;
  iSetup.get<IdealGeometryRecord>().get(pDD);
  edm::LogInfo("TkDetLayersAnalyzer") << " Top node is  " << &(*pDD) << "\n"
                                      << " And Contains  Daughters: " << (*pDD).components().size();

  ESHandle<TrackerTopology> tTopo_handle;
  iSetup.get<TrackerTopologyRcd>().get(tTopo_handle);
  const TrackerTopology* tTopo = tTopo_handle.product();

  /*

  // -------- here it constructs only a TOBLayer -------------------------
  vector<const GeometricDet*> geometricDetLayers = (*pDD).components();
  const GeometricDet* geometricDetTob=0;
  
  for(vector<const GeometricDet*>::const_iterator it=geometricDetLayers.begin();
      it!=geometricDetLayers.end(); it++){
    if(  (*it)->type() == GeometricDet::TOB ) {
      edm::LogInfo("TkDetLayersAnalyzer") << "found TOB geometricDet!" ;
      geometricDetTob = (*it);
    }
  }
  
  edm::LogInfo("TkDetLayersAnalyzer") << "Tob geometricDet has " << geometricDetTob->components().size() << " daughter" ;
  const GeometricDet* geometricDetTOBlayer = geometricDetTob->components()[1];

  edm::LogInfo("TkDetLayersAnalyzer") << "this Tob layer has: " << geometricDetTOBlayer->components().size() << " daughter" ;

  */

  /*
    vector<const GeometricDet*> geometricDetTOBlayer3Strings = geometricDetTOBlayer3->components();
    for(vector<const GeometricDet*>::const_iterator it=geometricDetTOBlayer3Strings.begin();
    it!=geometricDetTOBlayer3Strings.end(); it++){
    
    cout << "string phi position: " << (*it)->positionBounds().phi()  << endl;
    cout << "string r position:   " << (*it)->positionBounds().perp() << endl;
    cout << "string z position:   " << (*it)->positionBounds().z() << endl;
    cout << endl;
    }
  */

  /*
  TOBLayerBuilder myTOBBuilder;
  TOBLayer* testTOBLayer = myTOBBuilder.build(geometricDetTOBlayer,&(*pTrackerGeometry),&(*tTopo_handle));
  edm::LogInfo("TkDetLayersAnalyzer") << "testTOBLayer: " << testTOBLayer;

  */
  // ------------- END -------------------------

  //
  // -------------- example of using the Trie ---------------------------------
  //  code from GeometricSearchTrackerBuilder
  //

  using namespace trackerTrie;
  // create a Trie
  DetTrie trie(0);

  {
    const TrackingGeometry::DetContainer& modules = pTrackerGeometry->detUnits();
    typedef TrackingGeometry::DetContainer::const_iterator Iter;
    Iter b = modules.begin();
    Iter e = modules.end();
    Iter last;
    try {
      for (; b != e; ++b) {
        last = b;
        unsigned int rawid = (*b)->geographicalId().rawId();
        trie.insert(trackerHierarchy(tTopo, rawid), *b);
      }
    } catch (edm::Exception const& e) {
      std::cout << "in filling " << e.what() << std::endl;
      unsigned int rawid = (*last)->geographicalId().rawId();
      int subdetid = (*last)->geographicalId().subdetId();
      std::cout << rawid << " " << subdetid << std::endl;
    }
  }

  // layers "ids"
  unsigned int layerId[] = {1, 3, 5, 21, 22, 41, 42, 61, 62};

  // TOB is "2"
  {
    std::string s;
    if (layerId[2] > 9)
      s += char(layerId[2] / 10);
    s += char(layerId[2] % 10);
    node_iterator e;
    node_iterator tobl(trie.node(s));
    for (; tobl != e; tobl++) {
      // for each  tob layer and compute cylinder geom (not the real ones though)
      CylinderBuilderFromDet cylbld;
      WrapTrieCB<CylinderBuilderFromDet> w(cylbld);
      edm::iterateTrieLeaves(w, *tobl);
      std::unique_ptr<BoundCylinder> cyl(cylbld.build());
      SimpleCylinderBounds const& cylb = static_cast<SimpleCylinderBounds const&>(cyl->bounds());
      std::cout << "cyl " << tobl.label() << ": " << cylb.length() << ", " << cylb.width() << ", " << cylb.thickness()
                << std::endl;
    }
  }

  // ------------- END -------------------------

  // -------- here it constructs the whole GeometricSearchTracker --------------
  GeometricSearchTrackerBuilder myTrackerBuilder;
  GeometricSearchTracker* testTracker = myTrackerBuilder.build(&(*pDD), &(*pTrackerGeometry), &(*tTopo_handle));
  edm::LogInfo("TkDetLayersAnalyzer") << "testTracker: " << testTracker;

  for (auto const& l : testTracker->allLayers()) {
    auto const& layer = *l;
    std::cout << layer.seqNum() << ' ' << layer.subDetector() << ' ' << layer.basicComponents().size() << '\n';
    //auto mx = std::minmax_element (layer.basicComponents().begin(),layer.basicComponents().end(),[](    );
    auto m_min(std::numeric_limits<float>::max());
    auto m_max(std::numeric_limits<float>::min());
    for (auto const& c : layer.basicComponents()) {
      auto const& det = *c;
      auto xi = det.specificSurface().mediumProperties().xi();
      m_min = std::min(m_min, xi);
      m_max = std::max(m_max, xi);
      // std::cout <<  det.specificSurface().mediumProperties().xi() <<',';
    }
    std::cout << "xi " << m_min << '/' << m_max;
    std::cout << std::endl;
  }

  // ------------- END -------------------------
}

//define this as a plug-in
DEFINE_FWK_MODULE(TkDetLayersAnalyzer);
