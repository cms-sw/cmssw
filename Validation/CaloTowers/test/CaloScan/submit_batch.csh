#!/bin/csh
set i=1

while ( ${i} < 26 )
bsub -u /dev/null  -q 1nd batch.csh ${i}
@ i = ${i} + "1"
end
