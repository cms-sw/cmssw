import sys,os,commands

def main():
    if not sys.argv[0] or not sys.argv[1]:
        print "Usage: cpFromCastor fromDir toDir (optional runnumber)"
        exit(0)
    user = os.getenv("USER")
    castorDir = "/castor/cern.ch/cms/store/caf/user/" + user + "/" + sys.argv[1] + "/"
    aCommand = "nsls " + castorDir
    if sys.argv[3]:
        aCommand += " | grep " + sys.argv[3]
    output = commands.getstatusoutput(aCommand)
    if output[0] != 0:
        print output[1]
        exit(0)
    fileList = output[1].split('\n')
    cpCommand = "rfcp " + castorDir
    destDir = sys.argv[2]
    for file in fileList:
        aCommand = cpCommand + file + " " + destDir
        print ">>" + aCommand
        output = commands.getstatusoutput(aCommand)
        
if __name__ == "__main__":
    main()
