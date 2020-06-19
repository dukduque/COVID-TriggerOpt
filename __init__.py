# Clear temp files
import os
try:
    os.remove('./InterventionsMIP/temp/sir.ATTR')
except FileNotFoundError:
    print('No ATTR temp data')
try:
    os.remove('./InterventionsMIP/temp/sir.BAS')
except FileNotFoundError:
    print('No BAS temp data')