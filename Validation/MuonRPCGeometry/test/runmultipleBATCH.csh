#! /bin/csh


set scriptDir = `pwd`
#set script = testRPCTriggerEff.py
set script = testRPCTriggerEffFull.py

set cdir  = "$CASTOR_HOME/310/digiReal/"

mkdir out
rm -rf out/*

@ i = 0
foreach f (`nsls $cdir`) 

         set file = "rfio://$cdir$f"
        echo "   $file"
        bsub runsingle.csh $i $scriptDir $script $file
        @ i += 1
end


