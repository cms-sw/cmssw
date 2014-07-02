#include "SimCalorimetry/EcalSimAlgos/interface/EcalSignalGenerator.h"


  template <>
  CaloSamples EcalSignalGenerator<EBDigitizerTraits>::samplesInPE(const DIGI & digi)
  {
    // calibration, for future reference:  (same block for all Ecal types)
    //EcalDetId cell = digi.id();
    //         const EcalCalibrations& calibrations=conditions->getEcalCalibrations(cell);
    //const EcalQIECoder* channelCoder = theConditions->getEcalCoder (cell);
    //const EcalQIEShape* channelShape = theConditions->getEcalShape (cell);
    //EcalCoderDb coder (*channelCoder, *channelShape);
    //CaloSamples result;
    //coder.adc2fC(digi, result);
    //fC2pe(result);

    DetId detId = digi.id();

    double Emax = fullScaleEnergy(detId); 
    double LSB[NGAINS+1]; 

    //double icalconst = findIntercalibConstant( detId );

    double icalconst = 1.;  // find the correct value.

    const EcalIntercalibConstantMCMap &icalMap = ical->getMap();
    EcalIntercalibConstantMCMap::const_iterator icalit = icalMap.find(detId);
    if( icalit!=icalMap.end() )
    {
        icalconst = (*icalit);
    }

    double peToA = peToAConversion ( detId ) ;

    const std::vector<float> gainRatios = GetGainRatios(detId);

    for( unsigned int igain ( 0 ); igain <= NGAINS ; ++igain ) 
      {
	LSB[igain] = 0.;
	if ( igain > 0 ) LSB[igain]= Emax/(MAXADC*gainRatios[igain]);
      }

    //    std::cout << " intercal, LSBs, egains " << icalconst << " " << LSB[0] << " " << LSB[1] << " " << gainRatios[0] << " " << gainRatios[1] << " " << Emax << std::endl;

    CaloSamples result(detId, digi.size());

    for(int isample = 0; isample<digi.size(); ++isample){

      int gainId = digi[isample].gainId();
      //int gainId = 1;

      result[isample] = float(digi[isample].adc())*LSB[gainId]*icalconst/peToA;
    }

    //std::cout << " EcalSignalGenerator:EB noise input " << digi << std::endl;

    //std::cout << " converted noise sample " << std::endl;
    //for(int isample = 0; isample<digi.size(); ++isample){
    //  std::cout << " " << result[isample] ;
    //}
    //std::cout << std::endl;

    return result;
  }


  template <>
  CaloSamples EcalSignalGenerator<EEDigitizerTraits>::samplesInPE(const DIGI & digi)
  {
    // calibration, for future reference:  (same block for all Ecal types)
    //EcalDetId cell = digi.id();
    //         const EcalCalibrations& calibrations=conditions->getEcalCalibrations(cell);
    //const EcalQIECoder* channelCoder = theConditions->getEcalCoder (cell);
    //const EcalQIEShape* channelShape = theConditions->getEcalShape (cell);
    //EcalCoderDb coder (*channelCoder, *channelShape);
    //CaloSamples result;
    //coder.adc2fC(digi, result);
    //fC2pe(result);

    DetId detId = digi.id();

    double Emax = fullScaleEnergy(detId); 
    double LSB[NGAINS+1]; 

    double icalconst = 1.; //findIntercalibConstant( detId );

    const EcalIntercalibConstantMCMap &icalMap = ical->getMap();
    EcalIntercalibConstantMCMap::const_iterator icalit = icalMap.find(detId);
    if( icalit!=icalMap.end() )
    {
        icalconst = (*icalit);
    }


    double peToA = peToAConversion ( detId ) ;

    const std::vector<float> gainRatios = GetGainRatios(detId);

    for( unsigned int igain ( 0 ); igain <= NGAINS ; ++igain ) 
      {
	LSB[igain] = 0.;
	if ( igain > 0 ) LSB[igain]= Emax/(MAXADC*gainRatios[igain]);
      }

    //    std::cout << " intercal, LSBs, egains " << icalconst << " " << LSB[0] << " " << LSB[1] << " " << gainRatios[0] << " " << gainRatios[1] << " " << Emax << std::endl;

    CaloSamples result(detId, digi.size());

    for(int isample = 0; isample<digi.size(); ++isample){

      int gainId = digi[isample].gainId();
      //int gainId = 1;

      result[isample] = float(digi[isample].adc())*LSB[gainId]*icalconst/peToA;
    }

    //std::cout << " EcalSignalGenerator:EE noise input " << digi << std::endl;

    //std::cout << " converted noise sample " << std::endl;
    //for(int isample = 0; isample<digi.size(); ++isample){
    //  std::cout << " " << result[isample] ;
    // }
    //std::cout << std::endl;

    return result;
  }

  template <>
  CaloSamples EcalSignalGenerator<ESDigitizerTraits>::samplesInPE(const DIGI & digi)
  {
    // calibration, for future reference:  (same block for all Ecal types)
    //EcalDetId cell = digi.id();
    //         const EcalCalibrations& calibrations=conditions->getEcalCalibrations(cell);
    //const EcalQIECoder* channelCoder = theConditions->getEcalCoder (cell);
    //const EcalQIEShape* channelShape = theConditions->getEcalShape (cell);
    //EcalCoderDb coder (*channelCoder, *channelShape);
    //CaloSamples result;
    //coder.adc2fC(digi, result);
    //fC2pe(result);

    DetId detId = digi.id();

    double icalconst = 1.; //findIntercalibConstant( detId );

    const ESIntercalibConstantMap &icalMap = esmips->getMap();
    ESIntercalibConstantMap::const_iterator icalit = icalMap.find(detId);
    if( icalit!=icalMap.end() )
      {
	icalconst = double (*icalit);
      }

    CaloSamples result(detId, digi.size());

    for(int isample = 0; isample<digi.size(); ++isample){
      result[isample] = float(digi[isample].adc())/icalconst*ESMIPToGeV;
    }

    //std::cout << " EcalSignalGenerator:ES noise input " << digi << std::endl;

    //std::cout << " converted noise sample " << std::endl;
    //for(int isample = 0; isample<digi.size(); ++isample){
    //  std::cout << " " << result[isample] ;
    //}
    //std::cout << std::endl;

    return result;
  }

