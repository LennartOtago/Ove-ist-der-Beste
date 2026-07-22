import torch

def getMargfromSQTT(TTCore, Bounds, absError):
    """ Calculates Marginals for a TT-aproximation of the square root of a function.
        This works for simplest case: single layer DIRT (SIRT), with piecewise linear interpolation (basis = dt.Lagrange1(num_elems=29))
        get TTCores via:
        bridge = dt.SingleLayer()  # set single-layer DIRT (i.e., SIRT)
        # do DIRT (layered) as  : dirt = dt.DIRT(target_func, preconditioner, ftt)#, bridge) # do single-layer DIRT (i.e., SIRT)
        sirt_options = dt.DIRTOptions(num_error_samples=0)
        sirt = dt.DIRT(target_func, preconditioner, ftt ,bridge, sirt_options)

        TTCore = [None] * dim
        for i in range(0,dim):
            TTCore[i] =  sirt.sirts[0].ftt.tt.cores[i]

    :Reference: 
        Deep Composition of Tensor-Trains Using Squared Inverse Rosenblatt Transports
        Tiangang Cui & Sergey Dolgov
        https://link.springer.com/article/10.1007/s10208-021-09537-5


    :param TTCore: Tensor-Train from single layer SIRT
    
    :param Bounds: Boundaries of grid (dim x 2)

    :param absError: choose so that it is roughly smaller than L2 norm error, see gamma error variable in Eq. (19) Reference Paper Cui and Dolgov

    """  
    
    dim = len(TTCore)
    B = dim * [None]  # coeffTensor
    R = [None] * dim
    C = [None] * dim
    L = [None] * dim

    LebLam = 1  # !! Lebesgue Measure
    B[-1] = TTCore[-1]
    r_kmin1, n, r_k = TTCore[0].shape
    M = torch.eye(n) * (Bounds[0][1] - Bounds[0][0])  # Mass matrix
    L[0] = torch.linalg.cholesky(M)
    
    #backward marginalisation
    for k in range(dim - 1, 0, -1):
        #print(k)
        r_kmin1, n, r_k = TTCore[k].shape
        # !! we set Lebesgue Measure to const = one
        M = torch.eye(n) * (Bounds[k][1] - Bounds[k][0])  # Mass matrix
        L[k] = torch.linalg.cholesky(M)

        # construct Tensor C Eq. (27)
        C[k] = torch.einsum('ail,it->atl',  B[k], L[k])

        # unfold along first coordinate and compute thin QR decomposition of C^T
        # Eq. (28)
        Q, R[k] = torch.linalg.qr(torch.transpose(torch.reshape(C[k],(r_kmin1, n * r_k)) ,0, 1), mode='reduced') #, order='C'

        # compute next coefficient tensor
        # Eq. (29)
        B[k - 1] = torch.einsum('jbi,ki->jbk', TTCore[k - 1], R[k])

    #now pre compute marginal coefficients sarting at dim = 1, k = 0
    BPre = dim * [None]  # coeffTensor
    RPre = [None] * dim
    CPre = [None] * dim
    
    LebLam = 1  # !! Lebesgue Measure
    BPre[0] = TTCore[0]

    #forward marginalisation
    for k in range(0, dim-1):
        #print(k)
        r_kmin1, n, r_k = TTCore[k].shape
        # !! we set Lebesgue Measure to const = one
        #M = torch.eye(n) * (Bounds[k][1] - Bounds[k][0])  # Mass matrix
        #L = torch.linalg.cholesky(M)

        # construct Tensor C Eq. (27)
        CPre[k] = torch.einsum('ail,it->atl', BPre[k], L[k])

        # unfold along first coordinate and compute thin QR decomposition of C
        # Eq. (28)
        Q, RPre[k] = torch.linalg.qr(torch.reshape(CPre[k],(r_kmin1 * n, r_k)), mode='reduced') #, order='C')

        # compute next coefficient tensor
        # Eq. (29)
        BPre[k + 1] = torch.einsum('lj,jbi->lbi',  RPre[k],  TTCore[k + 1])


    # calculate marginal PDF
    margPDF = torch.zeros((dim, n))
    D = [None] * dim
    HoleLebLam = torch.zeros(dim)
    for k in range(0, dim):
        HoleLebLam[k] = (Bounds[k][1] - Bounds[k][0])

    gamError = absError/torch.prod(HoleLebLam)# (absError) ** 2 np.prod(HoleLebLam)
   
    for k in range(0, dim):
        if k == 0:
            #z = (absError) ** 2 + R[0][0][0] ** 2
            D[0] = B[0][0]
            # first marginal PDF
            # Eq. (30)
            margPDF[0] = (gamError * torch.prod(HoleLebLam[1:]) + torch.sum(D[0] ** 2, 1)) #/ z
            margPDF[0] = margPDF[0] / torch.sum(margPDF[0])
        elif k == dim-1:
            margPDF[k] = gamError * torch.prod(HoleLebLam[:k]) + torch.sum(BPre[k][:, :, 0] ** 2, 0) * HoleLebLam[k]
        else:
            # do other dimensions now
            D[k] = torch.einsum('lj,jbi->lbi', RPre[k-1],  B[k])
            margPDF[k] = gamError * torch.prod(HoleLebLam[k + 1:]) * torch.prod(HoleLebLam[:k]) + torch.sum(torch.sum( D[k] ** 2,0),1)  * HoleLebLam[k]
        margPDF[k] = margPDF[k] / torch.sum(margPDF[k])

    return margPDF
