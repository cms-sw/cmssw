/*
 Clustering algorithm "neighbor"
 Kristofer Henriksson
 
 This is a greedy clustering to be used for diagnostic purposes, which
 will make clusters as large as possible by including all contiguous hits
 in a single cluster.
*/

#ifndef CLUSTERING_ALGORITHM_neighbor_H
#define CLUSTERING_ALGORITHM_neighbor_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithmRecord.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

#include <boost/shared_ptr.hpp>

#include <string>
#include <cstdlib>
#include <map>

namespace cmsUpgrades{

template<typename T>
class ClusteringAlgorithm_neighbor : public ClusteringAlgorithm<T> {
public:
    typedef typename std::vector<T>::const_iterator inputIteratorType;
    typedef typename std::vector<T>::iterator mutInputIteratorType;

    ClusteringAlgorithm_neighbor(const StackedTrackerGeometry *i) :
        ClusteringAlgorithm<T>(i),
        mClassInfo(new classInfo(__PRETTY_FUNCTION__))
    {
    }
    
    ~ClusteringAlgorithm_neighbor()
    {
    }

    void Cluster(std::vector<std::vector<T> > &output,
                 const std::vector<T> &input) const;
    
    bool isANeighbor(const T& center, const T& mayNeigh) const;
    
    void addNeighbors(std::vector<T>& cluster, std::vector<T>& input) const;

    std::string AlgorithmName() const
    { 
        return mClassInfo->FunctionName() + "<" +
               mClassInfo->TemplateTypes().begin()->second + ">";
    }

private:
    const classInfo *mClassInfo;
};

template<typename T>
void ClusteringAlgorithm_neighbor<T>::Cluster(std::vector<std::vector<T> > &output,
                                              const std::vector<T> &input) const
{
	output.clear();
    std::vector<T> newInput = input;
    
    while (newInput.size() > 0) {
        std::vector<T> cluster;
        cluster.push_back(*newInput.begin());
        newInput.erase(newInput.begin());
        addNeighbors(cluster, newInput);
        output.push_back(cluster);
    }
}
    
template<typename T>
bool ClusteringAlgorithm_neighbor<T>::isANeighbor(const T& center,
                                                  const T& mayNeigh) const
{
    int rowdist = abs(center->row() - mayNeigh->row());
    int coldist = abs(center->column() - mayNeigh->column());
    return rowdist <= 1 && coldist <= 1;
}

template<typename T>
void ClusteringAlgorithm_neighbor<T>::addNeighbors(std::vector<T>& cluster,
                                                   std::vector<T>& input) const
{
    // This following line is necessary to ensure the iterators
    // afterward remain valid.
    cluster.reserve(input.size());
    mutInputIteratorType clus_it;
    mutInputIteratorType in_it;
    for (clus_it = cluster.begin(); clus_it < cluster.end(); clus_it++)
    {
        for (in_it = input.begin(); in_it < input.end(); in_it++)
        {
            if (isANeighbor(*clus_it, *in_it))
            {
                cluster.push_back(*in_it);
                in_it = input.erase(in_it) - 1;
            }
        }
    }
}

}


template<typename T>
class ES_ClusteringAlgorithm_neighbor: public edm::ESProducer{
public:
    ES_ClusteringAlgorithm_neighbor(const edm::ParameterSet & p)
    {
        setWhatProduced(this);
    }

    virtual ~ES_ClusteringAlgorithm_neighbor()
    {
    }

    boost::shared_ptr<cmsUpgrades::ClusteringAlgorithm<T> >
        produce(const cmsUpgrades::ClusteringAlgorithmRecord & record)
    { 
        edm::ESHandle<cmsUpgrades::StackedTrackerGeometry>
            StackedTrackerGeomHandle;
        record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get(StackedTrackerGeomHandle);
  
        cmsUpgrades::ClusteringAlgorithm<T>* ClusteringAlgo =
            new cmsUpgrades::ClusteringAlgorithm_neighbor<T>(&*StackedTrackerGeomHandle);

        _theAlgo = boost::shared_ptr<cmsUpgrades::ClusteringAlgorithm<T> >(ClusteringAlgo);

        return _theAlgo;
    } 

	private:
		boost::shared_ptr<cmsUpgrades::ClusteringAlgorithm<T> > _theAlgo;
};

#endif

