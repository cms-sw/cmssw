#!/bin/csh

    eval `scramv1 ru -csh`

    if ( -e outputHB.log ) rm outputHB.log
    if ( -e outputHE.log ) rm outputHE.log
    if ( -e outputHF.log ) rm outputHF.log
    
    cmsRun run_HB.cfg >& outputHB.log 
    cmsRun run_HE.cfg >& outputHE.log
    cmsRun run_HF.cfg >& outputHF.log

    root -b -q plot_HB.C
    root -b -q plot_HE.C
    root -b -q plot_HF.C
