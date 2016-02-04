#!/bin/csh
#The script should always be invoked with 1 argument: 
#the cpu core number 
#(0-4 on our lxbuild machines, 0-8 on dual quad-core machines).
#E.g (the ./ should not be necessary once this script is in CVS):
#taskset -c 0 ./cmsScimarkLaunch.csh 0
#or
#taskset -c 2 ./cmsScimarkLaunch.csh 2
#Set the environment
eval `scramv1 runtime -csh`
#Check if the cmsScimark output file exists: 
#if it does, move it to _old.log
if (-e cmsScimark_$1.log) then
mv cmsScimark_$1.log cmsScimark_$1_old.log
endif
#Get the date in the log for each iteration
date>cmsScimark_$1.log
set iterations=0
#Infinite loop of subsequent submissions of cmsScimark
while 1
#Script assumes cmsScimark2 command is in the release:
cmsScimark2>>cmsScimark_$1.log
#echo "$CMSSW_RELEASE_BASE/bin/slc4_ia32_gcc345/cmsScimark2>>cmsScimark_$1.log"
date>>cmsScimark_$1.log
#The following part was thought as a complement to this script:
#Automatically parse the logs and produce a Root plot 
#of the Composite Score vs time and of the Composite Score distribution
#in a html.
#Unfortunately it seems that using Root, after the number of Composite Scores 
#approaches ~2000, one starts to get segmentation faults if using the 
#SetBatch(1) option in PyRoot. This options avoids the annoying popping up of 
#X-windows (Root canvases) every time a plot is made. 
#As a consequence the SetBatch(1) option became SetBatch(0) by default in the 
#parsing/plotting script.
#So the solution for now is to exclude this parsing/plotting part from this 
#script, letting the user run it by hand when (s)he wants to see the results. 
#Alternatively one could uncomment the following and set a large number of 
#iterations at which to parse/publish the results, thus minimizing the 
#canvas popping... 
#Check if a cmsScimark results directory exists already for the wanted cpu:
#if not create it.
#if (!(-e cmsScimarkResults_cpu$1)) then
#mkdir cmsScimarkResults_cpu$1
#endif
#@ iterations = $iterations + 1
#if (!($iterations % 500)) then
#echo $iterations
#./cmsScimarkParser.py -i cmsScimark_$1.log -o cmsScimarkResults_cpu$1
#echo "./cmsScimarkParser.py -i cmsScimark_$1.log -o cmsScimarkResults_cpu$1"
#endif
end

