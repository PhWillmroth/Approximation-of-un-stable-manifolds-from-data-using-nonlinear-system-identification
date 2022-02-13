import pandas as pd
import skdim
from gpAlgo import gpAlgoTuned

liste = [
    ("linOscExact", "2D linear oscillator", 2), 
    ("cubOscDopri", "2D cubic oscillator", 2), 
    ("dddOscDopri", "3D linear oscillator", 3),
    ("lorenzDopri250", "3D Lorenz attractor", 3), # has corr. dim. 2.05 ± 0.01 as stated in [Grassberger and Procaccia 1983]
    ("doubPendDopri10", "2D double pendulum", 2)
]

for path, title, statedim in liste: 
    df = pd.read_csv( f'..\\data\\{path}.csv', index_col=0 )
    data = df.values[:, :statedim]
    
    cid = skdim.id.CorrInt(1, 100).fit(data)

    print(f"{title}: {cid.dimension_:.4f} / {statedim}")
    print(f"My implementation: {gpAlgoTuned(data, 200, True):.4f}")