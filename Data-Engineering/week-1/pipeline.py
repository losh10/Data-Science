import sys
import pandas as pd

print(sys.argv)        #PRINTS ALL PASSED ARGUMENTS
day = sys.argv[1]      

# some fancy stuff with pandas
print(f'job finished successfully for day = {day}')