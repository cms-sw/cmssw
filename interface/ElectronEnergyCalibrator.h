#ifndef ElectronEnergyCalibrator_H
#define ElectronEnergyCalibrator_H

#include "EgammaAnalysis/ElectronTools/interface/SimpleElectron.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

using std::string;
using std::vector;
using std::ifstream;
using std::istringstream;
using std::cout;

struct correctionValues
{
    double nRunMin;
    double nRunMax;
    double corrCat0;
    double corrCat1;
    double corrCat2;
    double corrCat3;
    double corrCat4;
    double corrCat5;
    double corrCat6;
    double corrCat7;
};

class ElectronEnergyCalibrator
{
    public:
        ElectronEnergyCalibrator( const std::string pathData, 
                                  const std::string dataset, 
                                  int correctionsType, 
                                  double lumiRatio, 
                                  bool isMC, 
                                  bool updateEnergyErrors, 
                                  bool verbose, 
                                  bool synchronization
                                ) : 
                                  pathData_(pathData), 
                                  dataset_(dataset), 
                                  correctionsType_(correctionsType), 
                                  lumiRatio_(lumiRatio), 
                                  isMC_(isMC), 
                                  updateEnergyErrors_(updateEnergyErrors), 
                                  verbose_(verbose), 
                                  synchronization_(synchronization) 
	    {
		    init();
    	}

        void calibrate(SimpleElectron &electron);

    private:
        void init();
        void splitString( const string &fullstr, 
                          vector<string> &elements, 
                          const string &delimiter
                        );
        double stringToDouble(const string &str);
      
        double newEnergy_ ;
        double newEnergyError_ ;
        
        std::string pathData_;
        std::string dataset_;
        int correctionsType_;
        double lumiRatio_;
        bool isMC_;
        bool updateEnergyErrors_;
        bool verbose_;
        bool synchronization_;
      
        correctionValues corrValArray[100];
        correctionValues corrValMC;
        int nCorrValRaw;
};

#endif

