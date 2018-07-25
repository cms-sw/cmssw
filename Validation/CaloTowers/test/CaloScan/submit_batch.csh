#!/bin/csh
set i=1

while ( ${i} < 51 )
bsub -u /dev/null  -q cmscaf1nd batch.csh ${i}
#bsub -u /dev/null  -q 1nd batch.csh ${i}
@ i = ${i} + "1"
end
