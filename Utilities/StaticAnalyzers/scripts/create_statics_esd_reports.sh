#!/usr/bin/env bash
export LC_ALL=C
eval `scram runtime -sh`
cd ${LOCALRT}/tmp

if [ ! -f ./function-calls-db.txt ]
	then
	echo "run ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/run_class_checker.sh first"
	exit 1
fi


if [ ! -f ./classes.txt.dumperft ]
        then
	echo "run ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/run_class_dumper.sh first"
	exit 2
fi


if [ ! -f ./data-class-funcs.py ]
	then 
	cp -pv ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/data-class-funcs.py .
fi
 
./data-class-funcs.py 2>&1 > data-class-funcs-report.txt
grep -v -e "^In call stack " data-class-funcs-report.txt | grep -v -e"Flagged event setup data class"  >override-flagged-classes.txt
grep -e "^Flagged event setup data class" data-class-funcs-report.txt | sort -u | awk '{print $0"\n\n"}' >esd2tlf.txt
grep -e "^In call stack" data-class-funcs-report.txt | sort -u | awk '{print $0"\n\n"}' >tlf2esd.txt

if [ ! -f ./statics.py ]
	then
	cp -pv ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/statics.py .
fi 
./statics.py 2>&1 > statics-report.txt.unsorted
sort -u < statics-report.txt.unsorted > statics-report.txt
grep -e "^In call stack " statics-report.txt | awk '{print $0"\n"}' > modules2statics.txt
grep -e "^Non-const static variable " statics-report.txt | awk '{print $0"\n"}' > statics2modules.txt

if [ ! -f ./edm-global-class.py ]
	then
	cp -pv ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/edm-global-class.py .
fi 

edm-global-class.py >edm-global-classes.txt.unsorted
sort -u edm-global-classes.txt.unsorted | grep -e"^EDM global class " | sort -u >edm-global-classes.txt
sort -u edm-global-classes.txt.unsorted | grep -v -e"^EDM global class " >edm-global-classes.txt.extra

if [ ! -f ./callgraph.py ]
   then
   cp -pv ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/callgraph.py .
   cp -pv ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/module_to_package.yaml .
   cp -pv ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/modules_in_ib.yaml .
   curl -OL https://raw.githubusercontent.com/fwyzard/circles/master/web/groups/packages.json
fi
touch eventsetuprecord-get-all.txt eventsetuprecord-get.txt
./callgraph.py 2>&1 | tee eventsetuprecord-get.txt
