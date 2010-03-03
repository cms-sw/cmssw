#ifndef L1TRIGGEROFFLINE_TRIGGERSIMULATION_HITMATCHINGALGORITHM_THRESHOLDS_H
#define L1TRIGGEROFFLINE_TRIGGERSIMULATION_HITMATCHINGALGORITHM_THRESHOLDS_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"
#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

namespace cmsUpgrades {
  
  /// hit position

  typedef Point2DBase< unsigned int,LocalTag > PixelLocation;
  
  template <class T> PixelLocation hitPosition(const T &hit)
    {
      return PixelLocation(hit->column(),hit->row());
    }

  /// Algorithm class

  template <class T> class HitMatchingAlgorithm_thresholds : public HitMatchingAlgorithm<T> {
    
  public:
    
    typedef std::vector<unsigned int> Thresholds;

    using HitMatchingAlgorithm<T>::theStackedTracker;
    
    HitMatchingAlgorithm_thresholds( const cmsUpgrades::StackedTrackerGeometry *i, const edm::ParameterSet& pset ) :
      // base
      HitMatchingAlgorithm<T>(i),
      // configuration
      vpset_(pset.getParameter< std::vector<edm::ParameterSet> >("Thresholds")),
      // class info
      info_( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__))
      {
      }
    
    ~HitMatchingAlgorithm_thresholds() 
      {
      }

    bool CheckTwoMemberHitsForCompatibility( const LocalStub<T> & localstub ) const
      {
	// const_cast of stub	
	LocalStub<T>& stub = const_cast< LocalStub<T>& >(localstub);
	
	// find hits -> 0 and 1 swapped due to bug in Geometry Builder ???
	// hit positions -> fix to allow for clusters
	PixelLocation inner = hitPosition(localstub.hit(1).at(0));
	PixelLocation outer = hitPosition(localstub.hit(0).at(0));

	// pixel geometry
	const GeomDet* geomdet = theStackedTracker->idToDet(stub.Id(),1);
	const PixelGeomDetUnit* pixeldet = dynamic_cast<const PixelGeomDetUnit*>(geomdet);
	const PixelTopology& pixeltopol = pixeldet->specificTopology();

	// hit position
	MeasurementPoint point(outer.y(),outer.x());
	GlobalPoint pos =  geomdet->surface().toGlobal(pixeltopol.localPosition(point));
	
	// layer number
	unsigned int layer = stub.Id().layer();
	
	std::vector<edm::ParameterSet>::const_iterator ipset = vpset_.begin();
	for (;ipset!=vpset_.end();ipset++) {
	  
	  // check layer
	  if (ipset->getParameter<unsigned int>("Layer")!=layer) continue;

	  // extract row threshold options
	  Thresholds rowcuts = ipset->getParameter< Thresholds >("RowCuts");
	  Thresholds rowoffsets = ipset->getParameter< Thresholds >("RowOffsets");
	  Thresholds rowwindows = ipset->getParameter< Thresholds >("RowWindows");

	  // find row cut
	  unsigned int i=0;
	  for (;i<rowcuts.size();i++) {if (outer.y()<rowcuts[i]) break;}

	  // set row thresholds
	  unsigned int rowoffset=rowoffsets[i]; 
	  unsigned int rowwindow=rowwindows[i];
	  
	  // set column thresholds
	  unsigned int columncut = (pos.eta()>0)?outer.x()-inner.x():inner.x()-outer.x();
	  unsigned int columnmin = ipset->getParameter<unsigned int>("ColumnCutMin");
	  unsigned int columnmax = ipset->getParameter<unsigned int>("ColumnCutMax");

	  // decision
	  bool row = (inner.y()-outer.y()-rowoffset>=0)&&(inner.y()-outer.y()-rowoffset<rowwindow);
	  bool col = (columncut>=columnmin)&&(columncut<=columnmax);

	  // return comparison	
	  return row&&col;
	}
	
	// if layer is not of interest return false
	return false;
      }
    
    /// algorithm name

    std::string AlgorithmName() const 
      { 
	return ((info_->FunctionName())+"<"+(info_->TemplateTypes().begin()->second)+">");
      }
    
  private:
    
    /// configurables
    std::vector<edm::ParameterSet> vpset_;  

    /// class info
    const cmsUpgrades::classInfo *info_;
 
  };
  
}

template <class T> class  ES_HitMatchingAlgorithm_thresholds : public edm::ESProducer {
  
 public:
  
  ES_HitMatchingAlgorithm_thresholds(const edm::ParameterSet & p) : 
    pset_(p) 
    {
      setWhatProduced( this );
    }
  
  virtual ~ES_HitMatchingAlgorithm_thresholds() 
    {
    }
  
  boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> > produce(const cmsUpgrades::HitMatchingAlgorithmRecord & record)
    { 
      // get record
      edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
      record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );

      // define algorithm
      algo_ = boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> >(new cmsUpgrades::HitMatchingAlgorithm_thresholds<T>( &(*StackedTrackerGeomHandle), pset_ ));

      // return algorithm
      return algo_;
    } 
  
 private:
  
  boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> > algo_;
  edm::ParameterSet pset_;
};

#endif

