
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef LOCAL_STUB_BUILDER_H
#define LOCAL_STUB_BUILDER_H

// system include files
#include <memory>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometryRecord.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetUnit.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithmRecord.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"


template<	typename T	>
class LocalStubBuilder : public edm::EDProducer {

	typedef std::vector<  cmsUpgrades::LocalStub< T > > LocalStubCollectionType;
    typedef std::pair<cmsUpgrades::StackedTrackerDetId,int> ClusterKey;
    typedef std::map<ClusterKey,std::vector<std::vector<T> > > ClusterMapType;

	public:
		explicit LocalStubBuilder(const edm::ParameterSet& iConfig);
		~LocalStubBuilder(){}

	private:
		virtual void beginJob(const edm::EventSetup& iSetup)
		{

			iSetup.get<cmsUpgrades::StackedTrackerGeometryRecord>().get(StackedTrackerGeomHandle);
			theStackedTrackers = StackedTrackerGeomHandle.product();

			iSetup.get<cmsUpgrades::ClusteringAlgorithmRecord>().get( clusteringAlgoHandle );
			iSetup.get<cmsUpgrades::HitMatchingAlgorithmRecord>().get( matchingAlgoHandle );

			std::cout	<< "LocalStubBuilder<" << (mClassInfo->TemplateTypes().begin()->second) << "> loaded modules:"
						<< "\n\tClusteringAlgorithm:\t"	<< clusteringAlgoHandle->AlgorithmName()
						<< "\n\tHitMatchingAlgorithm:\t"<< matchingAlgoHandle->AlgorithmName()
						<< std::endl;
		}

		void RetrieveRawHits( std::map< DetId, std::vector< T > > &mRawHits , const edm::Event& iEvent );



		virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
		{	
			typedef std::map< DetId, std::vector< T > > HitMapType;
            HitMapType rawHits;
			RetrieveRawHits( rawHits , iEvent );

			std::auto_ptr< LocalStubCollectionType > LocalStubsForOutput(new LocalStubCollectionType );
            
            std::auto_ptr<ClusterMapType> clusterPtr(new ClusterMapType);

			for (  StackedTrackerIterator = theStackedTrackers->stacks().begin(); StackedTrackerIterator != theStackedTrackers->stacks().end() ; ++StackedTrackerIterator ) {
                cmsUpgrades::StackedTrackerDetUnit* Unit = *StackedTrackerIterator;
                cmsUpgrades::StackedTrackerDetId Id = Unit->Id();
                
                assert(Unit == theStackedTrackers->idToStack(Id));

				std::vector< std::vector< T > > innerHits, outerHits;
				typedef typename std::vector< std::vector< T > >::iterator VVT_IT;
			
                typename HitMapType::const_iterator innerHitFind = rawHits.find(Unit->stackMember(0));
                typename HitMapType::const_iterator outerHitFind = rawHits.find(Unit->stackMember(1));
                
				if (innerHitFind != rawHits.end())
                    clusteringAlgoHandle->Cluster(innerHits, innerHitFind->second);
				if (outerHitFind != rawHits.end())
                    clusteringAlgoHandle->Cluster(outerHits, outerHitFind->second);
                
                if (storeClusters)
                {
                    if (innerHits.size())
                    {
                        ClusterKey inmapkey = std::make_pair(Id, 0);
                        (*clusterPtr)[inmapkey].insert((*clusterPtr)[inmapkey].end(),
                                                       innerHits.begin(),
                                                       innerHits.end());
                    }
                    if (outerHits.size())
                    {
                        ClusterKey outmapkey = std::make_pair(Id, 1);
                        (*clusterPtr)[outmapkey].insert((*clusterPtr)[outmapkey].end(),
                                                        outerHits.begin(),
                                                        outerHits.end());
                    }
                }

				if( innerHits.size() && outerHits.size() ){

					for( VVT_IT innerHitIter = innerHits.begin() ; innerHitIter != innerHits.end() ; ++innerHitIter ){
						for( VVT_IT outerHitIter = outerHits.begin() ; outerHitIter != outerHits.end() ; ++outerHitIter ){
							cmsUpgrades::LocalStub<T> tempLocalStub( Id );
							tempLocalStub.addCluster( 0 , *innerHitIter );
							tempLocalStub.addCluster( 1 , *outerHitIter );

							if( matchingAlgoHandle->CheckTwoMemberHitsForCompatibility( tempLocalStub ) ) LocalStubsForOutput->push_back( tempLocalStub );
						}
					}
				}
			}
			std::cout	<<"Made " << LocalStubsForOutput->size() << " local stubs of type " << (mClassInfo->TemplateTypes().begin()->second) << "." << std::endl;
			iEvent.put(LocalStubsForOutput);
            if (storeClusters)
            {
                std::string prodtype =
                    mClassInfo->TemplateTypes().begin()->second.substr(17);
                iEvent.put(clusterPtr, std::string("ClusteredHitsFrom") + prodtype);
            }
		}


		virtual void endJob(){}

     
		// ----------member data ---------------------------
		edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
		const cmsUpgrades::StackedTrackerGeometry *theStackedTrackers;
		cmsUpgrades::StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;

		edm::ESHandle< cmsUpgrades::ClusteringAlgorithm<T> > clusteringAlgoHandle;
		edm::ESHandle< cmsUpgrades::HitMatchingAlgorithm<T> > matchingAlgoHandle;

        std::vector<edm::InputTag>  rawHitInputTags;

		const cmsUpgrades::classInfo *mClassInfo;
        
        bool storeClusters;


		//For Digis
		unsigned int ADCThreshold;
};






///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Specialize the templates...
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template<typename T>
		LocalStubBuilder<T>::LocalStubBuilder<T>(const edm::ParameterSet& iConfig): mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) )
		{
			produces< LocalStubCollectionType >();
            
			rawHitInputTags  = iConfig.getParameter< std::vector<edm::InputTag> >("rawHits");
            storeClusters = iConfig.getParameter<bool>("storeClusters");
            if (storeClusters) {
                std::string prodtype =
                    mClassInfo->TemplateTypes().begin()->second.substr(17);
                produces<ClusterMapType>(std::string("ClusteredHitsFrom") +
                                                prodtype);
            }
		}

		template<>
		LocalStubBuilder<cmsUpgrades::Ref_PixelDigi_>::LocalStubBuilder<cmsUpgrades::Ref_PixelDigi_>(const edm::ParameterSet& iConfig): mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) )
		{
			produces< LocalStubCollectionType >();
			rawHitInputTags  = iConfig.getParameter< std::vector<edm::InputTag> >("rawHits");
            storeClusters = iConfig.getParameter<bool>("storeClusters");
			ADCThreshold = iConfig.getParameter<unsigned int>("ADCThreshold");
            if (storeClusters) {
                std::string prodtype =
                    mClassInfo->TemplateTypes().begin()->second.substr(17);
                produces<ClusterMapType>(std::string("ClusteredHitsFrom") +
                                                prodtype);
            }
		}

////////////////////////////////////////

		template<typename T>
		void 
		LocalStubBuilder<T>::RetrieveRawHits( std::map< DetId, std::vector<T> > &mRawHits , const edm::Event& iEvent)
		{
			mRawHits.clear();
		}

		template<>
		void 
		LocalStubBuilder<cmsUpgrades::Ref_TTHit_>::RetrieveRawHits( std::map< DetId, std::vector<cmsUpgrades::Ref_TTHit_> > &mRawHits , const edm::Event& iEvent)
		{
			mRawHits.clear();
			for(  std::vector<edm::InputTag>::iterator it = rawHitInputTags.begin(); it != rawHitInputTags.end(); ++it){
				edm::Handle< edm::DetSetVector< TrackTriggerHit > > HitHandle;
				iEvent.getByLabel( *it, HitHandle);

				edm::DetSetVector<TrackTriggerHit>::const_iterator	iter_over_dets;
				edm::DetSet<TrackTriggerHit>::const_iterator		iter_over_hits;

				for(iter_over_dets = HitHandle->begin(); iter_over_dets != HitHandle->end(); iter_over_dets++) {
					DetId id = iter_over_dets->id;
					if (id.subdetId()==1){
						for(iter_over_hits = iter_over_dets->data.begin(); iter_over_hits != iter_over_dets->data.end(); iter_over_hits++) {
							mRawHits[id].push_back( makeRefTo( HitHandle , id , iter_over_hits ) );
						}
					}	
				}
			}
		}

		template<>
		void 
		LocalStubBuilder<cmsUpgrades::Ref_PixelDigi_>::RetrieveRawHits( std::map< DetId, std::vector<cmsUpgrades::Ref_PixelDigi_> > &mRawHits , const edm::Event& iEvent)
		{
			mRawHits.clear();
			for(  std::vector<edm::InputTag>::iterator it = rawHitInputTags.begin(); it != rawHitInputTags.end(); ++it){
				edm::Handle< edm::DetSetVector< PixelDigi > > HitHandle;
				iEvent.getByLabel( *it, HitHandle);

				edm::DetSetVector<PixelDigi>::const_iterator iter_over_dets;
				edm::DetSet<PixelDigi>::const_iterator  iter_over_hits;

				for(iter_over_dets = HitHandle->begin(); iter_over_dets != HitHandle->end(); iter_over_dets++) {
					DetId id = iter_over_dets->id;
					if (id.subdetId()==1){
						for(iter_over_hits = iter_over_dets->data.begin(); iter_over_hits != iter_over_dets->data.end(); iter_over_hits++) {
							if( iter_over_hits->adc() >= ADCThreshold ) mRawHits[id].push_back( makeRefTo( HitHandle , id , iter_over_hits ) );
						}
					}	
				}
			}
		}


		template<>
		void
		LocalStubBuilder<cmsUpgrades::Ref_PSimHit_>::RetrieveRawHits( std::map< DetId, std::vector<cmsUpgrades::Ref_PSimHit_> > &mRawHits , const edm::Event& iEvent)
		{
			mRawHits.clear();
			for(  std::vector<edm::InputTag>::iterator it = rawHitInputTags.begin(); it != rawHitInputTags.end(); ++it){
				edm::Handle<edm::PSimHitContainer> HitHandle;
				iEvent.getByLabel( *it, HitHandle);

				for ( unsigned int i=0 ; i != HitHandle->size() ; ++i ) {
					DetId id = DetId(HitHandle->at(i).detUnitId());
					if (id.subdetId()==1) mRawHits[id].push_back( edm::Ref<edm::PSimHitContainer>( HitHandle , i ) );
				}
			}
		}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////






#endif

