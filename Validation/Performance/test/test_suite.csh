#!/bin/csh

    eval `scramv1 ru -csh`
    SealPluginRefresh
        
#    rm *.xml*
    if ( -e general.log ) rm general.log
    if ( -e timing.log )  rm timing.log

    cmsRun minbias_detsim_digi.cfg >& general.log
    
    grep TimeModule general.log >& timing.log

  
    root -b -q timing.C
    

