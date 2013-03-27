#ifndef SimpleElectron_H
#define SimpleElectron_H

class SimpleElectron
{
	public:
	SimpleElectron(){}
	SimpleElectron( double run, double eClass, double r9, double scEnergy, double scEnergyError, double trackMomentum, double trackMomentumError, double regEnergy, double regEnergyError, double eta, bool isEB, bool isMC) : run_(run),eClass_(eClass), r9_(r9),scEnergy_(scEnergy), scEnergyError_(scEnergyError), trackMomentum_(trackMomentum), trackMomentumError_(trackMomentumError), regEnergy_(regEnergy), regEnergyError_(regEnergyError), eta_(eta), isEB_(isEB), isMC_(isMC) {}
	~SimpleElectron(){}	
	//
	//accessors
	double getNewEnergy(){return newEnergy_;}
	double getNewEnergyError(){return newEnergyError_;}
	double getCombinedMomentum(){return combinedMomentum_;}
	double getCombinedMomentumError(){return combinedMomentumError_;}
	double getScale(){return scale_;}
	double getSmearing(){return smearing_;}
	double getSCEnergy(){return scEnergy_;}
	double getSCEnergyError(){return scEnergyError_;}
	double getRegEnergy(){return regEnergy_;}
	double getRegEnergyError(){return regEnergyError_;}
	double getTrackerMomentum(){return trackMomentum_;}
	double getTrackerMomentumError(){return trackMomentumError_;}
	double getEta(){return eta_;}
	float getR9(){return r9_;}
	int getElClass(){return eClass_;}
	int getRunNumber(){return run_;}
	bool isEB(){return isEB_;}
	bool isMC(){return isMC_;}

	//setters
	void setCombinedMomentum(double combinedMomentum){combinedMomentum_ = combinedMomentum;}
	void setCombinedMomentumError(double combinedMomentumError){combinedMomentumError_ = combinedMomentumError;}
	void setNewEnergy(double newEnergy){newEnergy_ = newEnergy;}
	void setNewEnergyError(double newEnergyError){newEnergyError_ = newEnergyError;}

	private:
	double run_, eClass_;
	double r9_;
	double scEnergy_, scEnergyError_, trackMomentum_, trackMomentumError_, regEnergy_, regEnergyError_;
	double eta_;
	bool isEB_;
	bool isMC_;
	double newEnergy_, newEnergyError_, combinedMomentum_, combinedMomentumError_;
	double scale_, smearing_;
};

#endif
