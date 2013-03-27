#ifndef ElectronEPcombinator_H
#define ElectronEPcombinator_H

#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include <stdio.h>
#include <math.h>

class ElectronEPcombinator
{
	public:
	ElectronEPcombinator(){} 
	void combine(SimpleElectron & electron); 
	private:
	SimpleElectron electron_;
	void computeEPcombination();
	double combinedMomentum_;
	double combinedMomentumError_;
	double scEnergy_, scEnergyError_, trackerMomentum_, trackerMomentumError_;
	int elClass_;

};

#endif
