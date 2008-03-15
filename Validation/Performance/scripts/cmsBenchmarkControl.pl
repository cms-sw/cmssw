#!/usr/bin/perl                                                                                                                               
#G.Benelli                                                                                                                                     
#Script to control the submission of cmsBenchmark.pl on multiple cores
#Get some environment variables to use                                                                                                          
$CMSSW_BASE=$ENV{'CMSSW_BASE'};
$CMSSW_RELEASE_BASE=$ENV{'CMSSW_RELEASE_BASE'};
$CMSSW_VERSION=$ENV{'CMSSW_VERSION'};
$HOST=$ENV{'HOST'};

if (@ARGV)
{
    $NumberOfCores=$ARGV[0];
}
else
{
    print "Please input the number of cpu cores on this machine!\n";
    print "Usage: cmsBenchmarkControl.pl 4\n";
    exit;
}
#To help performance reproducibility when running on 1 core:                                                                                   
#Submit executables only on core cpu1,                                                                                                         
#while running cmsScimark on the other cores                                                                                                   
#print "Submitting cmsScimarkLaunch to run on core cpu0\n";                                                                                    
#system("taskset -c 0 $PerformancePkg/scripts/cmsScimarkLaunch.csh 0&");                                                                       
#print "Submitting cmsScimarkLaunch to run on core cpu2\n";                                                                                    
#system("taskset -c 2 $PerformancePkg/scripts/cmsScimarkLaunch.csh 2&");                                                                       
#print "Submitting cmsScimarkLaunch to run on core cpu3\n";                                                                                    
#system("taskset -c 3 $PerformancePkg/scripts/cmsScimarkLaunch.csh 3&");                                                                       
#print "Submitting cmsScimarkLaunch to run on core cpu4\n";                                                                                    
#system("taskset -c 4 $PerformancePkg/scripts/cmsScimarkLaunch.csh 4&");                                                                       
#print "Submitting cmsScimarkLaunch to run on core cpu5\n";                                                                                    
#system("taskset -c 5 $PerformancePkg/scripts/cmsScimarkLaunch.csh 5&");                                                                       
#print "Submitting cmsScimarkLaunch to run on core cpu6\n";                                                                                    
#system("taskset -c 6 $PerformancePkg/scripts/cmsScimarkLaunch.csh 6&");                                                                       
#print "Submitting cmsScimarkLaunch to run on core cpu7\n";                                                                                    
#system("taskset -c 7 $PerformancePkg/scripts/cmsScimarkLaunch.csh 7&");  

for ($Core=0;$Core<$NumberOfCores;$Core++)
{
    print "Launching cmsBenchmark.pl $Core (i.e. on cpu core $Core)\n";
    $cmsBenchmarklog="cmsBenchmark_cpu".$Core.".log";
    $cmsBenchmark=`cmsBenchmark.pl $Core >& $cmsBenchmarklog &`;
    print $cmsBenchmark;
}
exit;
