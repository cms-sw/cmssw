#!/usr/bin/env python3

import sys
import json
import argparse

# exit codes
NO_CHANGES = 0
DATAFORMATS_CHANGED = 40
POLICY_VIOLATION = 41

def policyChecks(document):
    """Check policies on dictionary definitions. Return True if checks are fine."""
    # Contents to be added later
    return True

def updatePolicyChecks(reference, update):
    """Check policies on dictionary updates. Return True if checks are fine."""
    # Contents to be added later
    return True

def main(args):
    with open(args.baseline) as f:
        baseline = json.load(f)

    if args.pr is not None:
        with open(args.pr) as f:
            pr = json.load(f)
        pc1 = policyChecks(pr)
        if baseline != pr:
            pc2 = updatePolicyChecks(baseline, pr)
            if not (pc1 and pc2):
                return POLICY_VIOLATION

            print("Changes in persistable data formats")
            return DATAFORMATS_CHANGED
        if not pc1:
            return POLICY_VIOLATION
    else:
        if not policyChecks(baseline):
            return POLICY_VIOLATION

    return NO_CHANGES

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Check dictionary policies of the JSON output of edmDumpClassVersion. If one JSON document is given (--baseline; e.g. in IBs), only the dictionary definition policy checks are done. If two JSON documents are given (--baseline and --pr; e.g. in PR tests), the dictionary definition policy checks are done on the --pr document, and, in addition, if any persistable data formats are changed, additional checks are done to ensure the dictionary update is done properly. Exits with {NO_CHANGES} if there are no changes to persistent data formats. Exits with {DATAFORMATS_CHANGED} if persistent data formats are changed, and the update complies with data format policies. Exits with {POLICY_VIOLATION} if some data format policy is violated. Other exit codes (e.g. 1, 2) denote some failure in the script itself.")

    parser.add_argument("--baseline", required=True, type=str, help="JSON file for baseline")
    parser.add_argument("--pr", type=str, help="JSON file for baseline+PR")
    parser.add_argument("--transientDataFormatPackage", action="store_true", help="The JSON files are for a package that can have only transient data formats")

    args = parser.parse_args()
    sys.exit(main(args))
