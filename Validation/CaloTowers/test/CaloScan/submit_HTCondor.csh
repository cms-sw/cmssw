#!/bin/csh
set i=1

while ( ${i} < 51 )

#echo ' '
echo 'i='${i}   
#echo 'dir='${PWD}

condor_submit par1=${i} par2=${PWD} submit.sub
@ i = ${i} + "1"
end
