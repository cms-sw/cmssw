/**_________________________________________________________________
   class:   BooTools.cc
   package: TopPairBSM


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BooTools.cc,v 1.1.2.1 2009/03/08 03:26:22 yumiceva Exp $

________________________________________________________________**/


#include "TopQuarkAnalysis/TopPairBSM/interface/BooTools.h"

#include<iostream>

//_____________________________________________________________________________
BooTools::BooTools()
{
}

//_____________________________________________________________________________
BooTools::~BooTools()
{
}

//______________________________________________________________________________
double BooTools::fix4VectorsForMass( TLorentzVector &vec1, TLorentzVector &vec2,
									 double targetMass,
									 double upperwidth1, double upperwidth2,
									 double lowerwidth1, double lowerwidth2) {

	// code provided by Charles Plagger
	double mass = (vec1 + vec2).M();
	double factor = (targetMass / mass) - 1;
	bool isNeg = false;
	int sign = 1;
	double min = 0;
	double max = -1;
	double e1 = vec1.E();
	double e2 = vec2.E();
    double width1 = upperwidth1;
	double width2 = upperwidth2;
	// Do we need to move the reconstructed mass up or down to meet the
	// target mass?
	if (factor < 0)
	{
		// if we're here, we're making the reconstructed mass smaller
		isNeg = true;
		sign = -1;
		factor *= -1;
		// Switch widths if we can
		if (lowerwidth1 >= 0)
		{
			width1 = lowerwidth1;
			width2 = lowerwidth2;
		}
		// we now need to make sure that we don't let the factor get
		// so large (in the negative sense) that we let either of our
		// W daughters disappear
		max = e1 / width1;
		if (e2 / width2 < max)
		{
			max = e2 / width2;
		}
	}
	// get a starting point.
	double numSig = factor * (e1 + e2) / (width1 + width2);
	for (int loop = 1; loop < 20; ++loop)
	{
		TLorentzVector tempvec1 = vec1 * (1 + sign * numSig * width1 / e1);
		TLorentzVector tempvec2 = vec2 * (1 + sign * numSig * width2 / e2);
		double newMass = (tempvec1 + tempvec2).M();
		if (fabs(targetMass - newMass) < 0.001 )
		{
			break;
		}
		if ((newMass - targetMass) * sign > 0)
		{
			max = numSig;
			numSig = (min + max) / 2.;
		}  // if too much
		else 
		{
			// do we have a max defined?
			if (max < 0)
			{
				// no max definied
				min = numSig;
				numSig *= 1.2; // grow by 10%
			} else {
				min = numSig;
				numSig = (min + max) / 2.;
			}
		} // if not enough
	} // for loop
    // use latest numSig
	vec1 *= (1 + sign * numSig * width1 / e1);
	vec2 *= (1 + sign * numSig * width2 / e2);
	return sign * numSig;

}
