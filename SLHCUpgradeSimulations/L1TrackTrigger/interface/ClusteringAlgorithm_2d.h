
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef CLUSTERING_ALGORITHM_2d_H
#define CLUSTERING_ALGORITHM_2d_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithmRecord.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

#include <boost/shared_ptr.hpp>

#include <memory>
#include <sstream>
#include <string>
#include <bitset>

#include <map>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the algorithm is defined here...
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace cmsUpgrades{

template<typename T>
struct pixel_container{
	const T* centrePixel;

	std::bitset<8>	neighbours;
	bool kill0;
	bool kill1;
};


template<	typename T	>
class ClusteringAlgorithm_2d : public ClusteringAlgorithm<T> {
	public:
		typedef typename std::vector< T >::const_iterator inputIteratorType;

		//the member functions
		ClusteringAlgorithm_2d( const cmsUpgrades::StackedTrackerGeometry *i , bool aDoubleCountingTest) : 	ClusteringAlgorithm<T>( i ), 
																					mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ),
																					mDoubleCountingTest( aDoubleCountingTest ){}
		~ClusteringAlgorithm_2d() {}

		void Cluster( std::vector< std::vector< T > > &output , const std::vector< T > &input ) const;

		std::string AlgorithmName() const { 
			return ( (mClassInfo->FunctionName())+"<"+(mClassInfo->TemplateTypes().begin()->second)+">" );
		}

	private:
		const cmsUpgrades::classInfo *mClassInfo;
		bool mDoubleCountingTest;
};

template<typename T>
void ClusteringAlgorithm_2d< T >::Cluster( std::vector< std::vector< T > > &output , const std::vector< T > &input ) const {
	output.clear();

	std::map< std::pair< unsigned int , unsigned int> , pixel_container<T> > hitcontainer;
	typedef typename std::map<std::pair< unsigned int , unsigned int> , pixel_container<T> >::iterator LOCALMAPTYPEITERATOR;
	typedef typename std::vector< T >::const_iterator inputIteratorType;

//first fill all the hits to grid
	for( inputIteratorType inputIterator = input.begin(); inputIterator != input.end(); ++inputIterator ){
		hitcontainer[ std::make_pair(  (**inputIterator).row() , (**inputIterator).column() ) ].centrePixel = &(*inputIterator);
		hitcontainer[ std::make_pair(  (**inputIterator).row() , (**inputIterator).column() ) ].kill0=false;
		hitcontainer[ std::make_pair(  (**inputIterator).row() , (**inputIterator).column() ) ].kill1=false;
		hitcontainer[ std::make_pair(  (**inputIterator).row() , (**inputIterator).column() ) ].neighbours=0x00;
	}

//then search to see if neighbour hits exist
	for( LOCALMAPTYPEITERATOR centralpixel = hitcontainer.begin(); centralpixel != hitcontainer.end(); ++centralpixel){
		unsigned int	row	=	centralpixel	->	first.first;
		unsigned int	col	=	centralpixel	->	first.second;

		//considering grid:
		//		a	b	c	 	0	1	2					-->r/phi = increasing row
		//		d	x	e	=	3	x	4					|
		//		f	g	h		5	6	7					V	z = decreasing column

		//--------------------------------------------------------------------------------------------
			centralpixel->second.neighbours[0]	=	(	hitcontainer.find(	std::make_pair(	row-1	,	col+1	) )		!= hitcontainer.end()		);
			centralpixel->second.neighbours[1]	=	(	hitcontainer.find(	std::make_pair( row		,	col+1	) )		!= hitcontainer.end()		);
			centralpixel->second.neighbours[2]	=	(	hitcontainer.find(	std::make_pair( row+1	,	col+1	) )		!= hitcontainer.end()		);
		//--------------------------------------------------------------------------------------------
			centralpixel->second.neighbours[3]	=	(	hitcontainer.find(	std::make_pair( row-1	,	col		) )		!= hitcontainer.end()		);
		//	bool		x	= 	true;						 		//		row		,	col	
			centralpixel->second.neighbours[4]	=	(	hitcontainer.find(	std::make_pair(	row+1	,	col		) )		!= hitcontainer.end()		);
		//--------------------------------------------------------------------------------------------
			centralpixel->second.neighbours[5]	=	(	hitcontainer.find(	std::make_pair( row-1	,	col-1	) )		!= hitcontainer.end()		);
			centralpixel->second.neighbours[6]	=	(	hitcontainer.find(	std::make_pair( row		,	col-1	) )		!= hitcontainer.end()		);
			centralpixel->second.neighbours[7]	=	(	hitcontainer.find(	std::make_pair( row+1	,	col-1	) )		!= hitcontainer.end()		);
		//--------------------------------------------------------------------------------------------
	}

//then fill the kill bits
	for( LOCALMAPTYPEITERATOR centralpixel = hitcontainer.begin(); centralpixel != hitcontainer.end(); ++centralpixel){
		bool		adf	=	centralpixel->second.neighbours[0]	|	centralpixel->second.neighbours[3]	|	centralpixel->second.neighbours[5]	;
		bool		ceh	=	centralpixel->second.neighbours[2]	|	centralpixel->second.neighbours[4]	|	centralpixel->second.neighbours[7]	;

		centralpixel->second.kill0 = (		adf	&	ceh	);
		centralpixel->second.kill1 = (		adf	|	centralpixel->second.neighbours[6]	)	;
	}



	for( LOCALMAPTYPEITERATOR centralpixel = hitcontainer.begin(); centralpixel != hitcontainer.end(); ++centralpixel){
		unsigned int	row	=	centralpixel	->	first.first;
		unsigned int	col	=	centralpixel	->	first.second;

		bool kill2 = false;
		LOCALMAPTYPEITERATOR rhs;
		if	(	(rhs = hitcontainer.find(	std::make_pair(	row+1	,	col-1	) )	)	!= hitcontainer.end()		) kill2 |= rhs->second.kill0;
		if	(	(rhs = hitcontainer.find(	std::make_pair(	row+1	,	col		) )	)	!= hitcontainer.end()		) kill2 |= rhs->second.kill0;
		if	(	(rhs = hitcontainer.find(	std::make_pair(	row+1	,	col+1	) )	)	!= hitcontainer.end()		) kill2 |= rhs->second.kill0;

		if (	!centralpixel->second.kill0		&&	!centralpixel->second.kill1	&&	!kill2	){
			std::vector< T > temp;
			temp.push_back (	*hitcontainer[	std::make_pair(	row		,	col		) ].centrePixel	);
			if(	centralpixel->second.neighbours[0]	)	temp.push_back (	*hitcontainer[	std::make_pair(	row-1	,	col+1	) ].centrePixel	);
			if(	centralpixel->second.neighbours[1]	)	temp.push_back (	*hitcontainer[	std::make_pair( row		,	col+1	) ].centrePixel	);
			if(	centralpixel->second.neighbours[2]	)	temp.push_back (	*hitcontainer[	std::make_pair( row+1	,	col+1	) ].centrePixel );
			if(	centralpixel->second.neighbours[3]	)	temp.push_back (	*hitcontainer[	std::make_pair( row-1	,	col		) ].centrePixel );
			if(	centralpixel->second.neighbours[4]	)	temp.push_back (	*hitcontainer[	std::make_pair(	row+1	,	col		) ].centrePixel	);
			if(	centralpixel->second.neighbours[5]	)	temp.push_back (	*hitcontainer[	std::make_pair( row-1	,	col-1	) ].centrePixel	);
			if(	centralpixel->second.neighbours[6]	)	temp.push_back (	*hitcontainer[	std::make_pair( row		,	col-1	) ].centrePixel	);
			if(	centralpixel->second.neighbours[7]	)	temp.push_back (	*hitcontainer[	std::make_pair( row+1	,	col-1	) ].centrePixel	);
			output.push_back(temp);
		}
	}

// test for double counting!!!
	if(mDoubleCountingTest){
		std::set<std::pair< unsigned int , unsigned int> > test;
		std::set<std::pair< unsigned int , unsigned int> > doubles;
		typedef typename std::vector< std::vector< T > >::iterator outputIteratorType1;
		typedef typename std::vector< T >::iterator outputIteratorType2;
		for( outputIteratorType1 outputIterator1 = output.begin(); outputIterator1 != output.end() ; ++outputIterator1 ){
			for( outputIteratorType2 outputIterator2 = outputIterator1->begin(); outputIterator2 != outputIterator1->end() ; ++outputIterator2 ){
				if ( test.find(	std::make_pair(  (**outputIterator2).row() , (**outputIterator2).column()) ) != test.end() )
					doubles.insert( std::make_pair(  (**outputIterator2).row() , (**outputIterator2).column()) );
				else
					test.insert( std::make_pair(  (**outputIterator2).row() , (**outputIterator2).column()) );
			}
		}

		if(doubles.size()){
			std::set<std::pair< unsigned int , unsigned int> >::iterator it;
			std::stringstream errmsg;
			for(it=doubles.begin();it!=doubles.end();++it){
				errmsg<< "Double counted pixel: (" <<it->first<<","<<it->second<<")\n";
			}
			for( outputIteratorType1 outputIterator1 = output.begin(); outputIterator1 != output.end() ; ++outputIterator1 ){
				errmsg<<  "cluster: ";
				for( outputIteratorType2 outputIterator2 = outputIterator1->begin(); outputIterator2 != outputIterator1->end() ; ++outputIterator2 ){
					errmsg <<"| ("<<  (**outputIterator2).row() <<","<< (**outputIterator2).column()<< ") ";
				}
				errmsg<<  "|\n";
			}
			edm::LogError("ClusteringAlgorithm_2d") <<errmsg.str();
		}
	
	}


}




//For simhits, no clustering, just define a "cluster" of one hit for that hit
template<>
void ClusteringAlgorithm_2d<cmsUpgrades::Ref_PSimHit_>::Cluster( std::vector< std::vector<cmsUpgrades::Ref_PSimHit_> > &output , const std::vector<cmsUpgrades::Ref_PSimHit_> &input ) const {
	output.clear();
	std::vector< cmsUpgrades::Ref_PSimHit_ >::const_iterator inputIterator;
	for( inputIterator = input.begin(); inputIterator != input.end(); ++inputIterator ){
		std::vector< cmsUpgrades::Ref_PSimHit_ > temp;
		temp.push_back(*inputIterator);
		output.push_back(temp);
	}
}



}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ...and declared to the framework here
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<	typename T	>
class  ES_ClusteringAlgorithm_2d: public edm::ESProducer{
	public:
		ES_ClusteringAlgorithm_2d(const edm::ParameterSet & p) : mDoubleCountingTest( p.getParameter<bool>("DoubleCountingTest") ) {setWhatProduced( this );}

		virtual ~ES_ClusteringAlgorithm_2d() {}

		boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> > produce(const cmsUpgrades::ClusteringAlgorithmRecord & record)
		{ 
			edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
			record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
			cmsUpgrades::ClusteringAlgorithm<T>* ClusteringAlgo = new cmsUpgrades::ClusteringAlgorithm_2d<T>( &(*StackedTrackerGeomHandle), mDoubleCountingTest);

			_theAlgo  = boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> >( ClusteringAlgo );

			return _theAlgo;
		} 

	private:
		boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> > _theAlgo;
		bool mDoubleCountingTest;
};


#endif

