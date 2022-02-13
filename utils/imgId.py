#!/usr/bin/env python
import sys
sys.path.append( '.\\utils' )

def getImgId( selectedClass ):
    # read latest used imgId
    with open( '.\\plot\\id\\stand.txt', 'r' ) as stand:
        imgId = int( stand.read() ) + 1

    # save new imgId
    with open( '.\\plot\\id\\stand.txt', 'wt' ) as stand:
        stand.write( str(imgId) )   

    with open( '.\\plot\\id\\IdDict.txt', 'a' ) as out:
        attrs = vars(selectedClass).items()
        itemListString = f'\n#{imgId} {selectedClass.__name__}: '  # write id and class name
        itemListString += '; '.join( f"{item[0]}: {item[1]}" for item in attrs if item[0] not in ['__module__', '__doc__'] )
        out.write( itemListString  )

    return imgId
