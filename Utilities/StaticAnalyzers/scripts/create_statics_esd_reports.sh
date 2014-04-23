#!/usr/bin/env bash
export LC_ALL=C
eval `scram runtime -sh`
cd ${LOCALRT}/tmp

if [ ! -f ./db.txt ]
	then
	echo "run ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/run_class_checker.sh first"
	exit 1
fi


if [ ! -f ./classes.txt.dumperft ]
        then
	echo "run ${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/run_class_dumper.sh first"
	exit 2
fi

${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/statics.py 2>&1 > statics-report.txt.unsorted
sort -u < statics-report.txt.unsorted > statics-report.txt
awk -F\' 'NR==FNR{a[" "$0"::"]=1;next} {n=0;for(i in a){if(index($2,i) && $1 == "In call stack "){n=1}}} n' plugins.txt statics-report.txt | sort -u  | awk '{print $0"\n"}' > modules2statics.txt
awk -F\' 'NR==FNR{a[" "$0"::"]=1;next} {n=0;for(i in a){if(index($4,i) && $1 == "Non-const static variable "){n=1}}} n' plugins.txt statics-report.txt | sort -u  | awk '{print $0"\n"}' > statics2modules.txt

${CMSSW_BASE}/src/Utilities/StaticAnalyzers/scripts/data-class-funcs.py 2>&1 > data-class-funcs-report.txt
grep -v -e "^In call stack " data-class-funcs-report.txt | grep -v -e"Flagged event setup data class"| sort -u  >override-flagged-classes.txt
grep -e "^Flagged event setup data class" data-class-funcs-report.txt | sort -u | awk '{print $0"\n\n"}' >esd2tlf.txt
grep -e "^In call stack" data-class-funcs-report.txt | sort -u | awk '{print $0"\n\n"}' >tlf2esd.txt
