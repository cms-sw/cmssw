/*
 * CACell.h
 *
 *  Created on: Jan 29, 2016
 *      Author: fpantale
 */

#ifndef CACELL_H_
#define CACELL_H_

#include "RecHitsKDTree.h"

// tbb headers
#include <tbb/concurrent_vector.h>



class Cell
{
public:
	Cell() { }
	Cell(const RecHitsKDTree* hitsKDTree, int innerHitId, int outerHitId, int layerId) : theHitsKDTree(hitsKDTree), theCAState(0),
			theInnerHitId(innerHitId), theOuterHitId(outerHitId), theLayerId(layerId), hasFriends(false) {




	}

	bool areAlmostAligned(const Stub& hitA, const Stub& hitB, const Stub& hitC, const float epsilon)
	{
		double rA = sqrt(hitA.stub_x*hitA.stub_x + hitA.stub_y*hitA.stub_y);
		double rB = sqrt(hitB.stub_x*hitB.stub_x + hitB.stub_y*hitB.stub_y);
		double rC = sqrt(hitC.stub_x*hitC.stub_x + hitC.stub_y*hitC.stub_y);

		double zA = hitA.stub_z;
		double zB = hitB.stub_z;
		double zC = hitC.stub_z;

		return fabs((rA - rB) * (zA - zC) - (rA - rC) * (zA - zB)) <= epsilon;

	}
//TODO: move outside the Cell
	int neighborSearch(const tbb::concurrent_vector<int>& outerCells)
	{
		const float c_maxParAbsDifference[parNum]= {0.06, 0.07};


		int neighborNum = 0;

		for (auto i= 0; i < rightCells.thesize; ++i)
		{
			if(thecellsArray[rightCells.thedata[i]].theInnerHitId != theOuterHitId)
				continue;
			bool isNeighbor = true;

			isNeighbor = isNeighbor && (fabs((theparams.thedata[0] - thecellsArray[rightCells.thedata[i]].theparams.thedata[0]))  < c_maxParAbsDifference[0]);
			isNeighbor = isNeighbor && areAlmostAligned(thehitsArray[theInnerHitId], thehitsArray[theOuterHitId], thehitsArray[thecellsArray[rightCells.thedata[i]].theOuterHitId], 40);
			if(!isNeighbor)
				break;
			double delta = fabs((theparams.thedata[1] - thecellsArray[rightCells.thedata[i]].theparams.thedata[1]));
			double phiDistance=  delta< 0.5*two_pi ? delta : two_pi-delta;
			isNeighbor = isNeighbor && (phiDistance < c_maxParAbsDifference[1]);
			if(!isNeighbor)
				break;

			// if all the parameters are inside the range the right cell is a right neighbor.
			// viceversa this cell will be the left neighbors for rightNeighbor(i)
			if (isNeighbor)
			{
				thecellsArray[rightCells.thedata[i]].theInnerNeighbors.push_back(theId);
				theOuterNeighbors.push_back(thecellsArray[rightCells.thedata[i]].theId);
				++neighborNum;
			}

		}
		return neighborNum;
	}


	void isRootCell(tbb::concurrent_vector<int>* rootCells)
	{
		if(theInnerNeighbors.size()== 0 && theCAState >= 2)
		{
			rootCells->push_back(theId);
		}
	}

	// if there is at least one left neighbor with the same state (friend), the state has to be increased by 1.
	int getCAState  () const
	{
		return theCAState;
	}

	void evolve(const tbb::concurrent_vector<int>* cells) {
		hasFriends = false;
		for(auto i =0; i < theOuterNeighbors.size(); ++i)
		{
			if(cells->at(theOuterNeighbors.at(i)).getCAState() == theCAState)
			{
				hasFriends = true;
				break;
			}
		}
	}



	inline
	void hasSameStateNeighbor()
	{
		if(hasFriends)
		{
			theCAState++;
		}
	}


	//
	//
	//	//check whether a Cell and the root have compatible parameters.
	//	inline
	//	bool areCompatible(const Cell& a, const int innerTripletChargeHypothesis)
	//	{
	//
	//		return (a.thechargeHypothesis == innerTripletChargeHypothesis) || (a.thechargeHypothesis == 0) || (innerTripletChargeHypothesis ==0) ;
	//
	//	}
	//
	//


	// trying to free the track building process from hardcoded layers, leaving the visit of the graph
	// based on the neighborhood connections between cells.
	inline
	void findTracks ( tbb::concurrent_vector<CATrack>& foundTracks, const tbb::concurrent_vector<CACell>& cells, CATrack& tmpTrack) {

		// the building process for a track ends if:
		// it has no right neighbor
		// it has no compatible neighbor

		// the track is then saved if the number of hits it contains is greater than a threshold
		if(theOuterNeighbors.size() == 0 )
		{
			if( tmpTrack.size() >= c_minHitsPerTrack-1)
				foundTracks.push_back(tmpTrack);
			else
				return;
		}
		else
		{
			bool hasOneCompatibleNeighbor = false;
			for( auto i=0 ; i < theOuterNeighbors.size(); ++i)
			{
				if(tmpTrack.size() <= 2 || areCompatible(cells.at(theOuterNeighbors.at(i)), innermostTripletChargeHypothesis) )
				{
					hasOneCompatibleNeighbor = true;
					tmpTrack.push_back(theOuterNeighbors.at(i));
					cells.at(theOuterNeighbors.at(i)).findTracks(foundTracks,cells, tmpTrack );
					tmpTrack.pop_back();
				}
			}

			if (!hasOneCompatibleNeighbor && tmpTrack.size() >= c_minHitsPerTrack-1)
			{
				foundTracks.push(tmpTrack);
			}
		}

	}


	tbb::concurrent_vector<int> theInnerNeighbors;
	tbb::concurrent_vector<int> theOuterNeighbors;

	int theInnerHitId;
	int theOuterHitId;
	float theRadius;
	short int theLayerId;
	short int theCAState;
	bool isHighPtCell;
	bool hasFriends;
	RecHitsKDTree* theHitsKDTree;

};



#endif /* CMSSW_8_0_0_PRE4_SRC_RECOTRACKER_TKHITPAIRS_INTERFACE_CACELL_H_ */
