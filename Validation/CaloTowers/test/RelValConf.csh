#!/bin/tcsh

set VERSION=$1

cat run_onRelVal_cfg.py | sed -e s/HcalRecHitValidationRelVal.root/HcalRecHitValidationRelVal_TTbar_MC_${VERSION}.root/ | sed -e s/CaloTowersValidationRelVal.root/CaloTowersValidationRelVal_TTbar_MC_${VERSION}.root/ | sed -e s/NoiseRatesRelVal.root/NoiseRatesRelVal_TTbar_MC_${VERSION}.root/ > run_onRelVal_TTbar_MC_cfg.py

cat run_onRelVal_cfg.py | sed -e s/HcalRecHitValidationRelVal.root/HcalRecHitValidationRelVal_TTbar_Startup_${VERSION}.root/ | sed -e s/CaloTowersValidationRelVal.root/CaloTowersValidationRelVal_TTbar_Startup_${VERSION}.root/ | sed -e s/NoiseRatesRelVal.root/NoiseRatesRelVal_TTbar_Startup_${VERSION}.root/ > run_onRelVal_TTbar_Startup_cfg.py

cat run_onRelVal_cfg.py | sed -e s/HcalRecHitValidationRelVal.root/HcalRecHitValidationRelVal_QCD_MC_${VERSION}.root/ | sed -e s/CaloTowersValidationRelVal.root/CaloTowersValidationRelVal_QCD_MC_${VERSION}.root/ | sed -e s/NoiseRatesRelVal.root/NoiseRatesRelVal_QCD_MC_${VERSION}.root/ > run_onRelVal_QCD_MC_cfg.py

cat run_onRelVal_cfg.py | sed -e s/HcalRecHitValidationRelVal.root/HcalRecHitValidationRelVal_QCD_Startup_${VERSION}.root/ | sed -e s/CaloTowersValidationRelVal.root/CaloTowersValidationRelVal_QCD_Startup_${VERSION}.root/ | sed -e s/NoiseRatesRelVal.root/NoiseRatesRelVal_QCD_Startup_${VERSION}.root/ > run_onRelVal_QCD_Startup_cfg.py

cat run_onRelVal_cfg.py | sed -e s/HcalRecHitValidationRelVal.root/HcalRecHitValidationRelVal_HighPtQCD_MC_${VERSION}.root/ | sed -e s/CaloTowersValidationRelVal.root/CaloTowersValidationRelVal_HighPtQCD_MC_${VERSION}.root/ | sed -e s/NoiseRatesRelVal.root/NoiseRatesRelVal_HighPtQCD_MC_${VERSION}.root/ > run_onRelVal_HighPtQCD_MC_cfg.py
