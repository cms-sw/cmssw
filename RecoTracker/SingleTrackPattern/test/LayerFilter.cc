#include "RecoTracker/SingleTrackPattern/test/LayerFilter.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

LayerFilter::LayerFilter(const edm::ParameterSet& conf): conf_(conf){
	produces< edm::DetSetVector<SiStripCluster> > ();
	layers =conf_.getParameter<int>("Layer");
}

LayerFilter::~LayerFilter(){}

void LayerFilter::beginRun(edm::Run & run, const edm::EventSetup& es){
  ContTIB3Lay = 0;
  ContTIB3Mod = 0;
  ContTIB3Fil1 = 0;
  ContTIB3Fil2 = 0;
 
}

void LayerFilter::produce(edm::Event& e, const edm::EventSetup& c){
	//get inputs
	std::string stripClusterProducer = conf_.getParameter<std::string>("ClusterProducer");
   	edm::Handle<edm::DetSetVector<SiStripCluster> > clusterHandle;
   	e.getByLabel(stripClusterProducer, clusterHandle);
  	const edm::DetSetVector<SiStripCluster>* clusterCollection = clusterHandle.product();
	std::string cpe = conf_.getParameter<std::string>("StripCPE");
     	edm::ESHandle<StripClusterParameterEstimator> parameterestimator;
     	c.get<TrackerCPERecord>().get(cpe, parameterestimator); 
     	const StripClusterParameterEstimator &stripcpe(*parameterestimator);
	edm::ESHandle<TrackerGeometry> pDD;
     	c.get<TrackerDigiGeometryRecord>().get( pDD );
     	const TrackerGeometry &tracker(*pDD);	

	//ouptut collection
	std::vector< edm::DetSet<SiStripCluster> > vSiStripCluster;
	
	//loop on cluster to select the ones at positive y
	edm::DetSetVector<SiStripCluster>::const_iterator iDSV;
	for (iDSV = clusterCollection->begin(); iDSV != clusterCollection->end(); iDSV++){
	        unsigned int id = iDSV->id;
		DetId detId(id);
		SiStripDetId a(detId);

		int toblayer =0;
		int tiblayer =0;
		unsigned int TKlayers = 0;
		if (a.subdetId() == 3 ) {
		  tiblayer = TIBDetId(detId).layer();
		  TKlayers = tiblayer;
		}
		if (a.subdetId() == 5 ) {
		  toblayer = TOBDetId(detId).layer();
		  TKlayers = toblayer + 4;
		}
		

		const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)tracker.idToDetUnit(detId);
		edm::DetSet<SiStripCluster>::const_iterator iDS;
		edm::DetSet<SiStripCluster> outputDetSet;
		outputDetSet.id = id;
		
		for (iDS = iDSV->data.begin(); iDS != iDSV->data.end(); iDS++){
		  StripClusterParameterEstimator::LocalValues parameters=
		    stripcpe.localParameters(*iDS,*stripdet);
		  
		  if (TKlayers != layers  ) {   // Filter the layer under study. 
		    if (check(parameters.first, stripdet)) {
		      outputDetSet.data.push_back(*iDS);
		    }
		  }
		}
		if (outputDetSet.data.size()) vSiStripCluster.push_back(outputDetSet);
	}

	std::auto_ptr< edm::DetSetVector<SiStripCluster> > output(new edm::DetSetVector<SiStripCluster>(vSiStripCluster) );
	e.put(output);	
}

GlobalPoint LayerFilter::toGlobal(const LocalPoint& local, const StripGeomDetUnit* detunit){
	return detunit->surface().toGlobal(local);
}

bool LayerFilter::check(const LocalPoint& local, const StripGeomDetUnit* detunit){
	if (detunit->specificType().isBarrel()) return toGlobal(local, detunit).y()>0;
	else return true;
}
