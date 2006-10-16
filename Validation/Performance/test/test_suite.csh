#!/bin/csh

    eval `scramv1 ru -csh`
    SealPluginRefresh
        
    rm *.xml*
    if ( -e general.log ) rm general_minbias.log
    if ( -e timing.log )  rm timing_minbias.log
    

    cmsRun minbias_detsim_digi.cfg >& general_minbias.log
    
    cat /proc/cpuinfo | grep "cpu MHz" >& cpu_info.log

    grep TimeModule general_minbias.log >& timing_minbias.log


    if ( -e OscarProducer_minbias.ps ) rm OscarProducer_minbias.ps
    if ( -e Modules_minbias.ps ) rm Modules_minbias.ps
    
    root -b -q timing.C
    

