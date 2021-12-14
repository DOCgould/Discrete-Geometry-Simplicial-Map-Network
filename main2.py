import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from sklearn.datasets import load_digits


def barycentric(delaunay, points, k):  # k=id simplice
    n = len(points[0])
    b = delaunay.transform[k, :n].dot(
        np.transpose(points - delaunay.transform[k, n])
    )
    return np.c_[np.transpose(b), 1 - b.sum(axis=0)]


def cartesian(delaunay, barycentric, k):
    return barycentric.dot(delaunay.points[delaunay.simplices[k]])

def nn(dataset,p):
    """
    Desceription:
        An Algorithm that produces the Face of the Simplex

    """
    # the points
    data = dataset[0]

    # the classes
    classes = dataset[1]

    # the k-complex
    k = Delaunay(data)

    
    n = len(set(classes))

    c = np.zeros(n)

    # the l-complex

    #l = Delaunay(np.array([[0,0],[0,1],[1,0]]))
    ls = []
    n = len(set(classes))

    d=np.diag(np.ones(n))

    ls.append(np.zeros(n))

    for i in range(n):
        ls.append(d[i])

    print(ls)

    l = Delaunay(ls) 

    print("The Delauny Triangulation of the L - Subcomplex ")
    print(l)
    # the barycentric subdivisions of the points p in K
    b=barycentric(k,p,k.find_simplex(p)[0])

    print("The Barycentric Subdivisions of the Points p in K")
    print(b)

    image = classes[k.simplices[k.find_simplex(p)]][0]

    # We take the baricentric subdivisions with respect to one of the simplices
    # to which p belongs to positive barycentric coordinates
    # cartesian picture of p in l


    # The Mapping

    f=b.dot(l.points[image])
    # CARTESIANAS IMAGEN DE P EN L
    return  barycentric(l,f,0) # BARICÉNTRICAS DE P EN L


if __name__ == "__main__":
    from matplotlib.patches import Polygon

    dataset = (
        np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [-1, -1],
                [-1, 2],
                [2, 2],
                [-1, 2],
                [2, -1],
            ]
        ),
        np.array([0, 1, 1, 0, 2, 2, 2, 2, 2]),
    )

    #digits = load_digits()
    #fig, ax_array = plt.subplots(20, 20)
    #axes = ax_array.flatten()
    #for i, ax in enumerate(axes):
    #    ax.imshow(digits.images[i], cmap='gray_r')
    #plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    #plt.tight_layout(h_pad=0.5, w_pad=0.01)

    #X = np.loadtxt("embedding_digits.txt")
    X = dataset[0]
    p1 = [np.max(X[:,0])+1,np.max(X[:,1])+1]
    p2 = [np.max(X[:,0])+1,np.min(X[:,1])-1]
    p3 = [np.min(X[:,0])-1,np.min(X[:,1])-1]
    p4 = [np.min(X[:,0])-1,np.max(X[:,1])+1]

    #D = Polygon(np.array([p1,p2,p3,p4]), closed=False)
    #ax = plt.gca()
    #ax.add_patch(D)
    #plt.axis('scaled')
    #plt.show()

    X=np.concatenate((X,np.array([p1,p2,p3,p4])))
    y=np.concatenate((dataset[1],np.array([4,4,4,4])))

    dataset = (
        X,
        y
    )

    data = dataset[0]
    classes = dataset[1]

    k = Delaunay(data)
    plt.triplot(data[:, 0], data[:, 1], k.simplices)
    plt.scatter(data[:, 0], data[:, 1], c=classes)
    plt.show()


    vector_output = nn(dataset,[X[0]])
    print("Neural Network Output")
    print(vector_output)


    #print(nn(dataset,[X[1]]))
    #print(nn(dataset,[X[2]]))
    #print(nn(dataset,[X[3]]))
    #print(nn(dataset,[X[4]]))

