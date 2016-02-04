#!/bin/tcsh

set VERSION=$1

cat run_onRelVal_mc_cfg.py | sed -e s/HcalRecHitsHarvestingME.root/HcalRecHitValidationRelVal_TTbar_MC_${VERSION}.root/ | sed -e s/CaloTowersHarvestingME.root/CaloTowersValidationRelVal_TTbar_MC_${VERSION}.root/ | sed -e s/NoiseRatesHarvestingME.root/NoiseRatesRelVal_TTbar_MC_${VERSION}.root/ > run_onRelVal_TTbar_MC_cfg.py

cat run_onRelVal_startup_cfg.py | sed -e s/HcalRecHitsHarvestingME.root/HcalRecHitValidationRelVal_TTbar_Startup_${VERSION}.root/ | sed -e s/CaloTowersHarvestingME.root/CaloTowersValidationRelVal_TTbar_Startup_${VERSION}.root/ | sed -e s/NoiseRatesHarvestingME.root/NoiseRatesRelVal_TTbar_Startup_${VERSION}.root/ > run_onRelVal_TTbar_Startup_cfg.py

cat run_onRelVal_mc_cfg.py | sed -e s/HcalRecHitsHarvestingME.root/HcalRecHitValidationRelVal_QCD_MC_${VERSION}.root/ | sed -e s/CaloTowersHarvestingME.root/CaloTowersValidationRelVal_QCD_MC_${VERSION}.root/ | sed -e s/NoiseRatesHarvestingME.root/NoiseRatesRelVal_QCD_MC_${VERSION}.root/ > run_onRelVal_QCD_MC_cfg.py

cat run_onRelVal_startup_cfg.py | sed -e s/HcalRecHitsHarvestingME.root/HcalRecHitValidationRelVal_QCD_Startup_${VERSION}.root/ | sed -e s/CaloTowersHarvestingME.root/CaloTowersValidationRelVal_QCD_Startup_${VERSION}.root/ | sed -e s/NoiseRatesHarvestingME.root/NoiseRatesRelVal_QCD_Startup_${VERSION}.root/ > run_onRelVal_QCD_Startup_cfg.py

cat run_onRelVal_mc_cfg.py | sed -e s/HcalRecHitsHarvestingME.root/HcalRecHitValidationRelVal_HighPtQCD_MC_${VERSION}.root/ | sed -e s/CaloTowersHarvestingME.root/CaloTowersValidationRelVal_HighPtQCD_MC_${VERSION}.root/ | sed -e s/NoiseRatesHarvestingME.root/NoiseRatesRelVal_HighPtQCD_MC_${VERSION}.root/ > run_onRelVal_HighPtQCD_MC_cfg.py

cat run_onRelVal_startup_cfg.py | sed -e s/HcalRecHitsHarvestingME.root/HcalRecHitValidationRelVal_MinBias_Startup_${VERSION}.root/ | sed -e s/CaloTowersHarvestingME.root/CaloTowersValidationRelVal_MinBias_Startup_${VERSION}.root/ | sed -e s/NoiseRatesHarvestingME.root/NoiseRatesRelVal_MinBias_Startup_${VERSION}.root/ > run_onRelVal_MinBias_Startup_cfg.py
