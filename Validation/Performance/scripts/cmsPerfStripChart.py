#! /usr/bin/env python

import os, sys
try: import simplejson as json
except ImportError: import json


def operate(timelog, memlog, json_f, num):

    import re
    import commands
    import ROOT
    from datetime import datetime

    script_name= os.path.basename(__file__)

    # Open files and store the lines.
    timefile = open(timelog, 'r')
    timelog_lines=timefile.readlines()
    timefile.close()

    memfile = open(memlog, 'r')
    memlog_lines=memfile.readlines()
    memfile.close()
 
    # Get average and maximum rss.
    max_rss = average = ' '
    i=0
    while i < len(timelog_lines):
        line = timelog_lines[i]
        if 'Average Time' in line:
            line = line[:-1]
            line_list = line.split(' ')
            average = line_list[3]
            break
        i+=1
    i=0
    while i < len(memlog_lines):
        line = memlog_lines[i]
        if 'Maximum rss' in line:
            line=line[:-1]
            line_list = line.split(' ')
            max_rss = line_list[3]
            break
        i+=1

    # Get Integration Build's identifier
    IB = os.path.basename(commands.getoutput("echo $CMSSW_BASE"))

    # Check if log files were parsed properly and if the IB is valid using regular expressions.
    try: 
        # regex for a float
        regex = "^\d+\.?\d*$"
        if average == ' ' or re.match(regex, average) is None:
            raise RuntimeError('Could not parse \"' + timelog + '\" properly. Check if Average Time is defined correctly.')
        if max_rss == ' ' or re.match(regex, max_rss) is None:
            raise RuntimeError('Could not parse \"%s\" properly. Check if Maximum rss is defined correctly.' %memlog)
 
       # regex for dates 'YYYY-MM-DD-HHMM'
        regex = '(19|20|21)\d\d-(0[1-9]|1[012])-(0[1-9]|[12]'+\
                '[0-9]|3[01])-([01][0-9]|2[0-4])([0-5][0-9])$'
        if re.search(regex, IB) is None:
            raise RuntimeError('Not a valid IB.'+\
                               ' Valid IB [CMSSW_X_X_X_YYYY-MM-DD-HHMM]')
    except Exception, err:
        sys.stderr.write(script_name + ': Error: ' + str(err) + '\n')
        return 2
    
    # Open for getting the data.
    json_db = open(json_f, "r")
    dict = json.load(json_db)
    json_db.close()

    # Append new data before storing.
    ib_list = IB.split('_')
    cmsrelease = ib_list[0] + '_' + ib_list[1] + '_' + ib_list[2] + '_' + ib_list[3]
    data = {"IB" : ib_list[4], "average" : float(average), "max_rss" : float(max_rss)}
    dict["strips"].append(data)

    print 'Storing entry to \"' + json_f + '\" file with attribute values:\n' +\
          'IB=' + IB + '\naverage=' + average + '\nmax_rss=' + max_rss
    # Store the data in json file.
    json_db = open(json_f, "w+")
    json.dump(dict, json_db, indent=2)
    json_db.close()
    print 'File \"%s\" was updated successfully!' % json_f

    # Get IB datetime, in order to sort by it, the list of the entries.
    for record in dict["strips"]:
        time_list = record['IB'].split('-')
        d = datetime(int(time_list[0]), int(time_list[1]), int(time_list[2]), int(time_list[3][0:2]), int(time_list[3][2:]))
        record['IB'] = d

    # Sort the list.
    list = sorted(dict["strips"], key=lambda k : k['IB'])

    # Create the graph.
    ROOT.gROOT.SetStyle("Plain")
    outdir = '.'
    rootfilename=outdir + '/graph.root'
    myfile=ROOT.TFile(rootfilename, 'RECREATE')

    if num > len(dict["strips"]):
        new_num = len(dict['strips'])
        sys.stderr.write(script_name + ': Warning: There are less than ' + str(num) + ' entries in json file. Changed number to ' + str(new_num) + '.\n')
        num = new_num
  
    graph=ROOT.TGraph(num)
    graph.SetMarkerStyle(8)
    graph.SetMarkerSize(.7)
    graph.SetMarkerColor(1)
    graph.SetLineWidth(3)
    graph.SetLineColor(2)
    graph.SetTitle(cmsrelease)
    graph.SetName('AverageToRSS_graph')
    point_counter = 0   
    for i in range(num):
        # get date of IB
        datime = dict['strips'][i]['IB']
        year = int(datime.__format__("%Y"))
        month = int(datime.__format__("%m"))
        day = int(datime.__format__("%d"))
        hour = int(datime.__format__("%H"))
        min = int(datime.__format__("%M"))
        da = ROOT.TDatime(year, month, day, hour, min, 00 )

        # calculate the average/max_rss value
        average = float(dict['strips'][i]['average'])
        max_rss = float(dict['strips'][i]['max_rss'])
        value = average / max_rss

        graph.SetPoint(point_counter, da.Convert(), value)
        point_counter+=1
    graph.GetXaxis().SetTitle("Integration Build")
    graph.GetYaxis().SetTitleOffset(1.3)
    graph.GetYaxis().SetTitle("Average/rss")
    graph.GetXaxis().SetTimeDisplay(1)
    graph.GetXaxis().SetNdivisions(-503)
    graph.GetXaxis().SetTimeFormat("%Y-%m-%d %H:%M")
    graph.GetXaxis().SetTimeOffset(0, "gmt")
    mycanvas = ROOT.TCanvas('canvas')
    mycanvas.cd()
    graph.Draw("ALP")

    mycanvas.Print(outdir + "/averagetorss_graph.gif","gif")

    # write it on file
    graph.Write()
    mycanvas.Write()

   
########################################################################################### 

if __name__ == '__main__':

    import optparse, stat

    ################################
    # Definition of command usage. #
    ################################
    script_name= os.path.basename(__file__)
    usage = script_name + ' <options> -t TIMELOG -m MEMLOG'
    parser = optparse.OptionParser(usage)
    parser.add_option('-t', '--timelog',
                      action='store',
                      dest='timelog',
                      default='',
                      metavar='TIMELOG',
                      help='input file TIMELOG, the output of cmsTiming_parser.py')
    parser.add_option('-m', '--memlog',
                      action='store',
                      dest='memlog',
                      default='',
                      metavar='MEMLOG', 
                      help='input file MEMLOG, the output of cmsSimplememchecker_parser.py')
    parser.add_option('-j', '--jsonfile',
                     action='store',
                     dest='json_f',
                     default='strips.json',
                     metavar='FILE.JSON',
                     help='the .json file database')
    parser.add_option('-n', type='int',
                      action='store',
                      dest='num',
                      default='30',
                      metavar='NUM',
                      help='last NUM entries to be printed in the graph. Default is 30.')
    (options, args) = parser.parse_args()

    ######################################
    # Some error handling for the usage. #
    ######################################
    if options.timelog == '' or\
       options.memlog == '':
        sys.exit('%s: Missing file operands!\n' % script_name+\
                 'Type %s --help for more information!' % script_name)
    if not os.path.exists(options.timelog) or\
       not os.path.exists(options.memlog):
        sys.exit('%s: Error: Not present file(s)!' % script_name)

    #############################################
    # Validity of .json file-database.          #
    #############################################
 
    # The format that the json file must have:  
    format = "\n  {  \"strips\" :\n"   +\
             "    [\n      {\"IB\" : \"XXX_XXX\", \"average\" : M, \"max_rss\" : N},\n" +\
             "        .........................................\n" +\
             "    ]\n"+\
             "  }\n"

    json_db = open(options.json_f, "r+")
    try:
        # -check if the json file is empty; if yes, create a new database upon it
        if os.stat(options.json_f)[stat.ST_SIZE] == 0:
            sys.stderr.write(script_name + ': Warning: File \"' + options.json_f + '\" is empty. A new database will be created upon it.\n')
            json_db.write("{\n  \"strips\" : [\n  ]\n}\n")
            json_db.seek(0, 0)

        # -check if file loads as a valid json
        dict = json.load(json_db)

        # -check if strips key is there.(Look format above!)
        dict["strips"]

        # -check if value of strips is type of list
        if not isinstance(dict["strips"], list):
            print type(dict["status"])
            raise Exception

        # -check if the list has valid elements
        if dict["strips"]:
            for item in dict["strips"]:
                if not set(['IB', 'average', 'max_rss']).issubset(item):
                    raise KeyError
    except ValueError: 
        sys.exit(script_name + ': Error: Not a valid json file! Please, check the format:\n' + format)
    except KeyError:
        sys.exit(script_name + ': Error: Invalid format in the json file! Check it here:\n' + format)
    finally:
        json_db.close()

    ####################
    # Start operation. #
    ####################

    # sys.exit() used in order to return an exit code to shell, in case of error
    sys.exit(operate(options.timelog, options.memlog, options.json_f, options.num))


