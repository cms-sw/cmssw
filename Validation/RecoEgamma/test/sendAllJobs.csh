#!/bin/csh
set i=1
set N=4

while ( ${i} <= ${N} )
bsub -u /dev/null  -q 8nh batchValid.csh ${i}
@ i = ${i} + "1"
end

