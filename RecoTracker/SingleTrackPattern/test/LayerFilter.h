#ifndef LayerFilter_h
#define LayerFilter_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
//RecoLocalTracker
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
//Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"


class LayerFilter : public edm::EDProducer
{
	public:
	LayerFilter(const edm::ParameterSet& conf);	

	virtual ~LayerFilter();

    	virtual void beginRun(edm::Run & run, const edm::EventSetup& );

    	virtual void produce(edm::Event& e, const edm::EventSetup& c);

	private:
	unsigned int layers;
	edm::ParameterSet conf_;
	bool check(const LocalPoint& local, const StripGeomDetUnit* detunit);
	GlobalPoint toGlobal(const LocalPoint& local, const StripGeomDetUnit* detunit);

	int ContTIB3Mod,ContTIB3Fil1,ContTIB3Lay,ContTIB3Fil2; 
};



#endif
