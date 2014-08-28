import os, json, csv, sys
from ROOT import TFile, TNtuple

if len(sys.argv) < 2:
    print 'Usage: '+sys.argv[0]+' CONFFILE'
    print ' where CONFFILE is a JSON file with run and lumisection selection'
    print
    sys.exit()

jsonfile=sys.argv[1]
wdir = "lumis"

cmd=['lumiCalc.py -n 0.0429 -c frontier://LumiProd/CMS_LUMI_PROD -r ',' -o ','.csvt lumibyls']

a={}        
with open(jsonfile) as f:
    a = json.load(f)
    f.close()

if not os.path.isdir(wdir):
    os.system('mkdir '+wdir)

f = TFile(wdir+'/lumis.root','recreate')
ntuple = TNtuple('ntuple','data from ascii file','run:ls:lumiDelivered:lumiReported')

for run, lumis in a.iteritems():
    fullcmd=cmd[0]+run+cmd[1]+wdir+'/'+run+cmd[2]
    print 'Get luminosity information for run '+run
    os.system(fullcmd)
    rf=open(wdir+'/'+run+'.csvt','r')
    crf=csv.reader(rf)
    crf.next()
    for row in crf:
        ntuple.Fill(int(row[0]),int(row[1]),float(row[2]),float(row[3]))
    rf.close()
    os.system('rm '+wdir+'/'+run+'.csvt')

f.Write()
f.Close()
print 'Luminosity tree written to working directory ./'+wdir
os.system('cp '+jsonfile+' '+wdir+'/'+jsonfile)
sys.exit()
