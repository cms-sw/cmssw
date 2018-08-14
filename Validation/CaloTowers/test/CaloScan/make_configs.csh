#!/bin/csh
set template=template.py
set head=pi50

#--- service variables
set chunk = 1000
set     j = 1
set   one = 1


set i=1
while ( ${i} < 51)

if( ${i} < 51  ) then 
  echo "i :"  ${i}
   @ j =  ${i} - $one
   @ j =  $j * $chunk + $one
   echo "j :" ${j}
   cat ${template} | sed s/XXXXX/${j}/ >  ${head}_${i}.py
endif

@ i = ${i} + "1"
end
