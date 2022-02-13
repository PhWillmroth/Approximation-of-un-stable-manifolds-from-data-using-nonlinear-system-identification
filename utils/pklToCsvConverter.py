import sys
import pandas as pd

# use from utils folder via "py pklToCsvConverter.py <filenameHere>"

PATH = f'..\\data\\errordata_{ sys.argv[1] }'
df = pd.read_pickle( PATH + '.pkl' )

# Save as csv and print
pd.set_option('display.max_columns', None)
df.to_csv( PATH + '.csv' )
print( df )