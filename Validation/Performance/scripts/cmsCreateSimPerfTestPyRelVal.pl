#!/usr/bin/perl

#Get the CMSSW_VERSION variable to use
$CMSSW_VERSION=$ENV{'CMSSW_VERSION'};


#Default number of events for each set of tests:
$TimeSizeNumOfEvts=10;
$IgProfNumOfEvts=2;
$ValgrindNumOfEvts=1;

@Candle=(
    HiggsZZ4LM190, 
    MinBias,
    SingleElectronE1000, 
    SingleMuMinusPt1000, 
    SinglePiMinusPt1000, 
    TTbar, 
    ZPrimeJJM700
    );

%CmsDriverCandle=(
    $Candle[0]=>"\"HZZLLLL -e 190\"",
    $Candle[1]=>"\"MINBIAS\"",
    $Candle[2]=>"\"E -e 1000\"",
    $Candle[3]=>"\"MU- -e 1000\"",
    $Candle[4]=>"\"PI- -e 1000\"",
    $Candle[5]=>"\"TTBAR\"",
    $Candle[6]=>"\"ZPJJ\""
    );
%CmsDriverCandleNoBrackets=(
    $Candle[0]=>"HZZLLLL -e 190",
    $Candle[1]=>"MINBIAS",
    $Candle[2]=>"E -e 1000",
    $Candle[3]=>"MU- -e 1000",
    $Candle[4]=>"PI- -e 1000",
    $Candle[5]=>"TTBAR",
    $Candle[6]=>"ZPJJ"
    );
#Running TimingReport, TimeReport, SimpleMemoryCheck, EdmSize on all 7 candles
#With $TimeSizeNumOfEvts events each
foreach (@Candle)
{
   system(
	"mkdir "."$_"."_TimeSize;
	cd "."$_"."_TimeSize;
	cmsDriver.py $CmsDriverCandleNoBrackets{$_} -n $TimeSizeNumOfEvts --step=GEN --customise=Simulation.py >& "."$_"."_GEN.log;
	cmsSimPyRelVal.pl $TimeSizeNumOfEvts $CmsDriverCandle{$_} 0123;
	cmsRelvalreport.py -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& "."$_".".log;
	cd .."
	);
}

#Running IgProfPerf, IgProfMem (TOTAL, LIVE, ANALYSE) on $IgProfNumOfEvts ZPrimeJJ events
system(
    "mkdir ZPrimeJJM700_IgProf;
    cd ZPrimeJJM700_IgProf;
    cmsDriver.py $CmsDriverCandleNoBrackets{$_} -n $IgProfNumOfEvts --step=GEN --customise=Simulation.py >& ZPrimeJJM700_GEN.log;
    cmsSimPyRelVal.pl $IgProfNumOfEvts $CmsDriverCandle{$Candle[6]} 4567;
    cmsRelvalreport.py -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& ZPrimeJJM700.log;
    cd .."
    );

#Running ValgrindFCE callgrind and memcheck on $ValgrindNumOfEvts ZPrimeJJ event (DIGI only)
system(
    "mkdir ZPrimeJJM700_Valgrind;
    cd ZPrimeJJM700_Valgrind;
    cmsDriver.py $CmsDriverCandleNoBrackets{$Candle[6]} -n $ValgrindNumOfEvts --step=GEN --customise=Simulation.py >& ZPrimeJJM700_GEN.log;
    cmsSimPyRelVal.pl $ValgrindNumOfEvts "."$CmsDriverCandle{$Candle[6]}"." 89;grep -v SIM SimulationCandles_"."$CMSSW_VERSION".".txt \>tmp; 
    mv tmp SimulationCandles_"."$CMSSW_VERSION".".txt;
    cmsRelvalreport.py -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& ZPrimeJJM700.log;
    cd .."
    );

#Running ValgrindFCE callgrind and memcheck on $ValgrindNumOfEvts SingleMuMinus event (SIM only)
system(
    "mkdir SingleMuMinusPt1000_Valgrind;
    cd SingleMuMinusPt1000_Valgrind;
    cmsDriver.py $CmsDriverCandleNoBrackets{$Candle[3]} -n $ValgrindNumOfEvts --step=GEN --customise=Simulation.py >& SingleMuMinusPt1000_GEN.log
    cmsSimPyRelVal.pl $ValgrindNumOfEvts "."$CmsDriverCandle{$Candle[3]}"." 89;grep -v DIGI SimulationCandles_"."$CMSSW_VERSION".".txt \>tmp; 
    mv tmp SimulationCandles_"."$CMSSW_VERSION".".txt;
    cmsRelvalreport.py -i SimulationCandles_"."$CMSSW_VERSION".".txt -t perfreport_tmp -R -P >& SingleMuMinusPt1000.log;
    cd .."
    );
exit;

