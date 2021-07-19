#!/usr/bin/env python

import argparse
import os
import subprocess


def main() :
    
    # Argument parser
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    
    parser.add_argument(
        "--sfind",
        help = "Find string",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--srepl",
        help = "Replace string",
        type = str,
        required = True,
    )
    
    parser.add_argument(
        "--dirs",
        help = "Directories to search in",
        type = str,
        nargs = "*",
        required = False,
        default = ["./"],
    )
    
    parser.add_argument(
        "--doreplace",
        help = "Perform the replace operation",
        default = False,
        action = "store_true",
    )
    
    args = parser.parse_args()
    d_args = vars(args)
    
    
    dirs = " ".join(args.dirs)
    
    cmd = "grep -Irl --exclude \"*.sh\" %s %s | sort -V" %(args.sfind, dirs)
    print(cmd)
    cmd_output = subprocess.check_output(cmd, shell = True)
    #print(cmd_output)
    
    l_fileName = cmd_output.strip().split()
    l_fileName = [fName.strip() for fName in l_fileName]
    l_fileName = [fName for fName in l_fileName if len(fName)]
    #print(l_fileName)
    
    nFile = len(l_fileName)
    
    print("\n")
    print("Replace: %s --> %s" %(args.sfind, args.srepl))
    print("\n")
    
    for iFile, fName in enumerate(l_fileName) :
        
        print("File (%d/%d): %s" %(iFile+1, nFile, fName))
        
        cmd = "sed -i \"s#%s#%s#g\" %s" %(args.sfind, args.srepl, fName)
        print(cmd)
        
        
        if (args.doreplace) :
            
            print ("Starting replace operation...")
            
            cmd_ret = os.system(cmd)
            
            if (cmd_ret) :
                
                print("Error.")
                exit(1)
            
            print ("Finished replace operation.")
        
        else :
            
            print("Not performing replace operation.")
        
        
        print("")
    
    
    
    return 0


if (__name__ == "__main__") :
    
    main()
