#include "SimpleNavigationSchool.h"

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

//class FakeDetLayer;


/** Concrete navigation school for cosmics in the Tracker
 */

class dso_hidden CosmicNavigationSchool : public SimpleNavigationSchool {
public:
  CosmicNavigationSchool(const GeometricSearchTracker* theTracker,
			 const MagneticField* field);
  ~CosmicNavigationSchool() override{ cleanMemory();}

  class CosmicNavigationSchoolConfiguration{
  public:
    CosmicNavigationSchoolConfiguration() : noPXB(false), noPXF(false), noTOB(false), noTIB(false), noTEC(false), noTID(false) , self(false), allSelf(false) {}
    CosmicNavigationSchoolConfiguration(const edm::ParameterSet& conf);
    bool noPXB;
    bool noPXF;
    bool noTOB;
    bool noTIB;
    bool noTEC;
    bool noTID;
    
    bool self;
    bool allSelf;
  };

  void build(const GeometricSearchTracker* theTracker,
	     const MagneticField* field,
	     const CosmicNavigationSchoolConfiguration conf);
 
protected:
  CosmicNavigationSchool(){}
private:

  //FakeDetLayer* theFakeDetLayer;
  void linkBarrelLayers( SymmetricLayerFinder& symFinder) override;
  //void linkForwardLayers( SymmetricLayerFinder& symFinder); 
  using SimpleNavigationSchool::establishInverseRelations;
  void establishInverseRelations( SymmetricLayerFinder& symFinder );
  void buildAdditionalBarrelLinks();
  void buildAdditionalForwardLinks(SymmetricLayerFinder& symFinder);
};


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimpleBarrelNavigableLayer.h"
#include "SimpleForwardNavigableLayer.h"
#include "SimpleNavigableLayer.h"
#include "DiskLessInnerRadius.h"
#include "SymmetricLayerFinder.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/src/DetBelowZ.h"
#include "TrackingTools/DetLayers/src/DetLessZ.h"

#include <functional>
#include <algorithm>
#include <map>
#include <cmath>

using namespace std;

CosmicNavigationSchool::CosmicNavigationSchoolConfiguration::CosmicNavigationSchoolConfiguration(const edm::ParameterSet& conf){
  noPXB=conf.getParameter<bool>("noPXB");
  noPXF=conf.getParameter<bool>("noPXF");
  noTIB=conf.getParameter<bool>("noTIB");
  noTID=conf.getParameter<bool>("noTID");
  noTOB=conf.getParameter<bool>("noTOB");
  noTEC=conf.getParameter<bool>("noTEC");
  self = conf.getParameter<bool>("selfSearch");
  allSelf = conf.getParameter<bool>("allSelf");
}

CosmicNavigationSchool::CosmicNavigationSchool(const GeometricSearchTracker* theInputTracker,
					       const MagneticField* field)
{
  build(theInputTracker, field, CosmicNavigationSchoolConfiguration());
}
 
void CosmicNavigationSchool::build(const GeometricSearchTracker* theInputTracker,
				   const MagneticField* field,
				   const CosmicNavigationSchoolConfiguration conf)
{
  LogTrace("CosmicNavigationSchool") << "*********Running CosmicNavigationSchool***********" ;	
  theBarrelLength = 0;theField = field; theTracker = theInputTracker;

  theAllDetLayersInSystem=&theInputTracker->allLayers();
  theAllNavigableLayer.resize(theInputTracker->allLayers().size(),nullptr);




  // Get barrel layers
  vector<BarrelDetLayer const*> const& blc = theTracker->barrelLayers();
  for ( auto i = blc.begin(); i != blc.end(); i++) {
    if (conf.noPXB && GeomDetEnumerators::isTrackerPixel((*i)->subDetector())) continue;
    if (conf.noTOB && (*i)->subDetector() == GeomDetEnumerators::TOB) continue;
    if (conf.noTIB && (*i)->subDetector() == GeomDetEnumerators::TIB) continue;
    theBarrelLayers.push_back( (*i) );
  }

  // get forward layers
  vector<ForwardDetLayer const*> const& flc = theTracker->forwardLayers();
  for ( auto i = flc.begin(); i != flc.end(); i++) {
    if (conf.noPXF && GeomDetEnumerators::isTrackerPixel((*i)->subDetector())) continue;
    if (conf.noTEC && (*i)->subDetector() == GeomDetEnumerators::TEC) continue;
    if (conf.noTID && (*i)->subDetector() == GeomDetEnumerators::TID) continue;
    theForwardLayers.push_back( (*i) );
  }

  FDLI middle = find_if( theForwardLayers.begin(), theForwardLayers.end(),
                         not1(DetBelowZ(0)));
  theLeftLayers  = FDLC( theForwardLayers.begin(), middle);
  theRightLayers = FDLC( middle, theForwardLayers.end());

  SymmetricLayerFinder symFinder( theForwardLayers);

  // only work on positive Z side; negative by mirror symmetry later
  linkBarrelLayers( symFinder);
  linkForwardLayers( symFinder);
  establishInverseRelations( symFinder );

  if (conf.self){

    // set the self search by hand
    //   NavigationSetter setter(*this);

    //add TOB1->TOB1 inward link
    const std::vector< const BarrelDetLayer * > &  tobL = theInputTracker->tobLayers();
    if (!tobL.empty()){
      if (conf.allSelf){
	LogDebug("CosmicNavigationSchool")<<" adding all TOB self search.";
	for (auto lIt = tobL.begin(); lIt!=tobL.end(); ++lIt)
	  dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[(*lIt)->seqNum()])->theSelfSearch = true;
      }else{
	SimpleNavigableLayer* navigableLayer = 
	  dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[tobL.front()->seqNum()]);
	LogDebug("CosmicNavigationSchool")<<" adding TOB1 to TOB1.";
	navigableLayer->theSelfSearch = true;
      }
    }
    const std::vector< const BarrelDetLayer * > &  tibL = theInputTracker->tibLayers();
    if (!tibL.empty()){
      if (conf.allSelf){
	LogDebug("CosmicNavigationSchool")<<" adding all TIB self search.";
	for (auto lIt = tibL.begin(); lIt!=tibL.end(); ++lIt)
	  dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[(*lIt)->seqNum()])->theSelfSearch = true;
      }else{
	SimpleNavigableLayer* navigableLayer = 
	      dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[tibL.front()->seqNum()]);
	LogDebug("CosmicNavigationSchool")<<" adding tib1 to tib1.";
	navigableLayer->theSelfSearch = true;
      }
    }
    const std::vector< const BarrelDetLayer * > &  pxbL = theInputTracker->pixelBarrelLayers();
    if (!pxbL.empty()){
      if (conf.allSelf){
	LogDebug("CosmicNavigationSchool")<<" adding all PXB self search.";
        for (auto lIt = pxbL.begin(); lIt!=pxbL.end(); ++lIt)
          dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[(*lIt)->seqNum()])->theSelfSearch = true;
      }else{
	SimpleNavigableLayer* navigableLayer =
	  dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[pxbL.front()->seqNum()]);
	LogDebug("CosmicNavigationSchool")<<" adding pxb1 to pxb1.";
	navigableLayer->theSelfSearch = true;
      }
    }
  }
}

void CosmicNavigationSchool::
linkBarrelLayers( SymmetricLayerFinder& symFinder)
{
  //identical to the SimpleNavigationSchool one, but it allows crossing over the tracker
  //is some non-standard link is needed, it should probably be added here
  
  // Link barrel layers outwards
  for ( BDLI i = theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
    BDLC reachableBL;
    FDLC leftFL;
    FDLC rightFL;

    // always add next barrel layer first
    if ( i+1 != theBarrelLayers.end()) reachableBL.push_back(*(i+1));

    // Add closest reachable forward layer (except for last BarrelLayer)
    if (i != theBarrelLayers.end() - 1) {
      linkNextForwardLayer( *i, rightFL);
    }

    // Add next BarrelLayer with length larger than the current BL
    if ( i+2 < theBarrelLayers.end()) {
      linkNextLargerLayer( i, theBarrelLayers.end(), reachableBL);
    }

    theBarrelNLC.push_back( new
       SimpleBarrelNavigableLayer( *i, reachableBL,
                                   symFinder.mirror(rightFL),
                                   rightFL,theField, 5.,false));
  }
}

// identical to  SimpleNavigationSchool but for the last additional stuff
void CosmicNavigationSchool::establishInverseRelations(SymmetricLayerFinder& symFinder) {
    
    //again: standard part is identical to SimpleNavigationSchool one. 
    //After the standard link, special outsideIn links are added  

    // NavigationSetter setter(*this);

    setState(navigableLayers());


    // find for each layer which are the barrel and forward
    // layers that point to it
    typedef map<const DetLayer*, vector<const BarrelDetLayer*>, less<const DetLayer*> > BarrelMapType;
    typedef map<const DetLayer*, vector<const ForwardDetLayer*>, less<const DetLayer*> > ForwardMapType;


    BarrelMapType reachedBarrelLayersMap;
    ForwardMapType reachedForwardLayersMap;

    for ( auto bli :  theBarrelLayers) {
      auto reachedLC = nextLayers(*bli, insideOut);
      for ( auto i : reachedLC) {
        reachedBarrelLayersMap[i].push_back(bli);
      }
    }

    for ( auto fli : theForwardLayers) {
      auto reachedLC = nextLayers(*fli, insideOut);
      for ( auto i : reachedLC) {
        reachedForwardLayersMap[i].push_back(fli);
      }
    }

     for(auto nl : theAllNavigableLayer) {
      if (!nl) continue;
      auto navigableLayer = static_cast<SimpleNavigableLayer*>(nl);
      auto dl = nl->detLayer();
      navigableLayer->setInwardLinks( reachedBarrelLayersMap[dl],reachedForwardLayersMap[dl] );
    }

   //buildAdditionalBarrelLinks();
    buildAdditionalForwardLinks(symFinder); 

}


void CosmicNavigationSchool::buildAdditionalBarrelLinks(){
  for (auto i = theBarrelLayers.begin(); i != theBarrelLayers.end(); i++) {
    SimpleNavigableLayer* navigableLayer =
      dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[(*i)->seqNum()]);
    if (i+1 != theBarrelLayers.end() )navigableLayer->setAdditionalLink(*(i+1), outsideIn);
  }
}


void CosmicNavigationSchool::buildAdditionalForwardLinks(SymmetricLayerFinder& symFinder){
  //the first layer of FPIX should not check the crossing side (since there are no inner layers to be tryed first)
  SimpleNavigableLayer* firstR = 
    dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[theRightLayers.front()->seqNum()]);
  SimpleNavigableLayer* firstL = 
    dynamic_cast<SimpleNavigableLayer*>(theAllNavigableLayer[theLeftLayers.front()->seqNum()]);
  firstR->setCheckCrossingSide(false);	
  firstL->setCheckCrossingSide(false);	
    	
  for (auto i : theRightLayers){
    //look for first bigger barrel layer and link to it outsideIn
    SimpleForwardNavigableLayer*  nfl = 
      dynamic_cast<SimpleForwardNavigableLayer*>(theAllNavigableLayer[(i)->seqNum()]);
    SimpleForwardNavigableLayer* mnfl = 
      dynamic_cast<SimpleForwardNavigableLayer*>(theAllNavigableLayer[symFinder.mirror(i)->seqNum()]);
    for (auto j : theBarrelLayers) {
      if ((i)->specificSurface().outerRadius() < (j)->specificSurface().radius() && 
	  fabs((i)->specificSurface().position().z()) < (j)->surface().bounds().length()/2.){ 
	nfl ->setAdditionalLink(j, outsideIn);
	mnfl->setAdditionalLink(j, outsideIn);	
	break;
      }	
    }
  }	  	
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

 
#include "NavigationSchoolFactory.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
DEFINE_EDM_PLUGIN(NavigationSchoolFactory, CosmicNavigationSchool, "CosmicNavigationSchool");




#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include <vector>

//class FakeDetLayer;


/** Concrete navigation school for cosmics in the Tracker
 */

class SkippingLayerCosmicNavigationSchool : public CosmicNavigationSchool {
public:
  SkippingLayerCosmicNavigationSchool(const GeometricSearchTracker* theTracker,
				      const MagneticField* field,
				      const CosmicNavigationSchoolConfiguration conf);

  ~SkippingLayerCosmicNavigationSchool() override{cleanMemory();};
};


SkippingLayerCosmicNavigationSchool::SkippingLayerCosmicNavigationSchool(const GeometricSearchTracker* theInputTracker,
									 const MagneticField* field,
									 const CosmicNavigationSchoolConfiguration conf)
{
  build(theInputTracker, field, conf);
}




#include <FWCore/Utilities/interface/ESInputTag.h>

#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "NavigationSchoolFactory.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

//
// class decleration
//
class dso_hidden SkippingLayerCosmicNavigationSchoolESProducer final : public edm::ESProducer {
 public:
  SkippingLayerCosmicNavigationSchoolESProducer(const edm::ParameterSet& iConfig) {
    theNavigationPSet = iConfig;
    theNavigationSchoolName = theNavigationPSet.getParameter<std::string>("ComponentName");
    //the following line is needed to tell the framework what
    // data is being produced
    setWhatProduced(this, theNavigationSchoolName);
  }

  ~SkippingLayerCosmicNavigationSchoolESProducer() override{}

  using ReturnType = std::unique_ptr<NavigationSchool>;

  ReturnType produce(const NavigationSchoolRecord&);

  // ----------member data ---------------------------
  edm::ParameterSet theNavigationPSet;
  std::string theNavigationSchoolName;
};

SkippingLayerCosmicNavigationSchoolESProducer::ReturnType
SkippingLayerCosmicNavigationSchoolESProducer::produce(const NavigationSchoolRecord& iRecord) {

  std::unique_ptr<NavigationSchool> theNavigationSchool ;

  // get the field
  edm::ESHandle<MagneticField>                field;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(field);

  //get the geometricsearch tracker geometry
  edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;
  iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);

  CosmicNavigationSchool::CosmicNavigationSchoolConfiguration layerConfig(theNavigationPSet);
  theNavigationSchool.reset(new SkippingLayerCosmicNavigationSchool(geometricSearchTracker.product(), field.product(), layerConfig) );

  return theNavigationSchool;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_EVENTSETUP_MODULE(SkippingLayerCosmicNavigationSchoolESProducer);


