import pandas as pd
from sklearn import manifold as mf

methodDict = {
        'LLE' : lambda comp, neigh: mf.LocallyLinearEmbedding( n_components=comp, n_neighbors=neigh ),
        'MLLE' : lambda comp, neigh: mf.LocallyLinearEmbedding( n_components=comp, n_neighbors=neigh, method='modified' ),
        'HLLE' : lambda comp, neigh: mf.LocallyLinearEmbedding( n_components=comp, n_neighbors=neigh, method='hessian' ),
        'LTSA' : lambda comp, neigh: mf.LocallyLinearEmbedding( n_components=comp, n_neighbors=neigh, method='ltsa' ),
        'Isomap' : lambda comp, neigh: mf.Isomap( n_components=comp, n_neighbors=neigh ),
        'SpectralEmbedding' : lambda comp, neigh: mf.SpectralEmbedding( n_components=comp, n_neighbors=neigh ),
        'MDS' : lambda comp, neigh: mf.MDS( n_components=comp ),
        'TSNE' : lambda comp, neigh: mf.TSNE( n_components=comp )
        }

def reduceDimension( self, inputData ):
    while True:
        try:
            embedding = methodDict[self.method]( self.intrinsicDim, self.nNeighbours ) # set up embedding
            transfData = embedding.fit_transform( inputData ) # embed
            break
        except:
            print( f'Increase number of neighbours to {sp.nNeighbours+1}.' )
            sp.nNeighbours += 1 # increase number of neighbours

    return pd.DataFrame( transfData, inputData.index, ['z'] ) # data, index, columnnames
