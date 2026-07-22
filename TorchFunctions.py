import numpy as np
import scipy as scy
import pandas as pd
from scipy import constants
import time
import torch
from torch import Tensor

def torch_pressure_to_height(p0, pplus, x, h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0):
    R = constants.gas_constant
    R_Earth = 6356#6371  # earth radiusin km6356#
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    #temp = get_temp(x)
    temp = torch_temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0)
    dP = pplus - p0
    #return - np.log(p0/pplus) /(-28.97 * grav / R /temp )
    #return ( np.log(pplus) -np.log(p0))/ (-28.97 * grav / R / temp)
    M = 28.97
    return (dP/p0) /(-M * grav / R /temp ), temp


def torch_height_to_pressure(x, dx, p0,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0):
    R = constants.gas_constant
    R_Earth = 6356#6371  # earth radiusin km6356#
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = torch_temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0)
    #dP = pplus - p0
    M = 28.97
    return dx * (-M * grav / R /temp ) * p0, temp


def torch_temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,a5,a6,b0):
    a = torch.ones(x.shape)
    b = torch.ones(x.shape)

    a[x < h0] = a0
    a[h0 <= x] = a1
    a[h1 <= x] = a2
    a[h2 <= x] = a3
    a[h3 <= x] = a4
    a[h4 <= x ] = a5
    a[h5 <= x ] = a6
    #a[h6 <= x ] = 0

    b[x < h0] = b0
    b[h0 <= x] = b0 + h0 * a0
    b[h1 <= x] = b0 + (h1 - h0) * a1 + h0 * a0
    b[h2 <= x] = a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h3 <= x ] = a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h4 <= x ] = a4 * (h4 -h3) + a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h5 <= x ] = a5 * (h5 -h4) + a4 * (h4 -h3) + a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    #b[h6 <= x ] = a4 * (h6-h5) + a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0


    h = torch.ones(x.shape)
    h[x < h0] = 0
    h[h0 <= x] = h0
    h[h1 <= x] = h1
    h[h2 <= x] = h2
    h[h3 <= x] = h3
    h[h4 <= x] = h4
    h[h5 <= x] = h5
    #h[h6 <= x] = h6
    return a * (x - h) + b



def get_height_temp_from_press(df_pres_val,tempParams):
    n = df_pres_val.shape[0]
    # set pressure at h = 0 to mean sea level pressure
    df_pres_val[0] = 1013.25
    df_hei_val = torch.zeros(df_pres_val.shape[0])
    
    df_temp_val = torch.zeros(df_pres_val.shape[0])
    df_temp_val[0] = 288.15
    
    for i in range(1,df_pres_val.shape[0]):
        dx, df_temp_val[i-1] = torch_pressure_to_height(df_pres_val[i-1], df_pres_val[i], df_hei_val[i-1], *tempParams)
        df_hei_val[i] = df_hei_val[i - 1] + dx
    
    return df_hei_val, df_temp_val


def get_press_temp_from_height(hei_val, min_press, tempParams):
    n = hei_val.shape[0]
    pres_val = torch.zeros(n)
    temp_val = torch.zeros(n)
    pres_val[0] =  min_press
    
    
    for i in range(1, n):
        dx = hei_val[i - 1] - hei_val[i]
        dp, temp_val[i-1] =  torch_height_to_pressure(hei_val[i - 1], dx, pres_val[i - 1], *tempParams )
        pres_val[i] = pres_val[i - 1] - dp
    temp_val[-1] = torch_temp_func(hei_val[-1], *tempParams)
    
    return pres_val, temp_val

def load_hitran_param(files):
    my_data = pd.read_csv(files, header=None)
    data_set = my_data.values
    
    size = data_set.shape
    wvnmbr = torch.zeros((size[0],1))
    S = torch.zeros((size[0],1))
    F = torch.zeros((size[0],1))
    g_air = torch.zeros((size[0],1))
    g_self = torch.zeros((size[0],1))
    E = torch.zeros((size[0],1))
    n_air = torch.zeros((size[0],1))
    g_doub_prime= torch.zeros((size[0],1))
    g_prime= torch.zeros((size[0],1))
    
    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
        S[i] = float(lines[0][16:25]) # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][148:153])
        g_prime[i] = float(lines[0][155:160])
    
    


    return wvnmbr, S, F, g_air, g_self, E, n_air, g_doub_prime, g_prime
    
''' generate dx'''
def torch_gen_forward_map(meas_ang, heights, obs_height, R):
    tang_height = (torch.sin(meas_ang) * (obs_height + R)) - R
    num_meas = len(tang_height)
    A_height = torch.zeros((num_meas, len(heights)-1))

    for m in range(0, num_meas):
        t = 0
        #find t so that layers[t] is larger than tang height
        while heights[t] < tang_height[m]:
            t += 1

        for i in range(t, len(heights)):
            A_height[m, i-1] = torch.sqrt((heights[i] + R) ** 2 - (tang_height[m] + R) ** 2) - torch.sum( A_height[m, :i])

    return A_height, tang_height, heights[-1]


def torch_gen_sing_map(dxs, tang_heights, heights):
    m,n = dxs.shape
    A_lin = torch.zeros((m,n+1))
    for i in range(0,m):
        t = 0
        while heights[t] <= tang_heights[i]:
            t += 1
        #print(t)
        #print(torch_gen_trap_rul(dxs[i, t - 1:]).shape)
        #print(   A_lin[i, t - 1:].shape)
        A_lin[i, t - 1:] = torch_gen_trap_rul(dxs[i, t - 1:])
        # A_lin[i, t-1] = 0.5 * dxs[i, t-1]
        # for j in range(t, n):
        #     A_lin[i,j] = 0.5 * (dxs[i,j-1] + dxs[i,j])
        # A_lin[i, -1] = 0.5 * dxs[i, -1]
    return A_lin


def torch_gen_trap_rul(dxs):
    #val = np.zeros(len(dxs)+1)
    #print(len(dxs))
    sumMat = torch.eye(len(dxs)+1)
    #print(sumMat)
    Ones = torch.ones((len(dxs)+1,len(dxs)+1))
    sumMat = sumMat + torch.triu(Ones,1) - torch.triu(Ones,2)
    return 0.5*(dxs @ torch.clone(sumMat[:-1,:]))

def torch_composeAforO3(A_lin, temp, press, ind, wvnmbr, g_doub_prime, g_prime, E, S):
        # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0] # in cm^-1


    Q = g_doub_prime[ind, 0] * torch.exp(- HitrConst2 * E[ind, 0] / temp) + g_prime[ind, 0] * torch.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / temp)
    Q_ref = g_doub_prime[ind, 0] * torch.exp(- HitrConst2 * E[ind, 0] / 296) + g_prime[ind, 0] * torch.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / 296)
    LineIntScal = Q_ref / Q * torch.exp(- HitrConst2 * E[ind, 0] / temp) / torch.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - torch.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - torch.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3
    C2 = constants.h * constants.c * v_0 * 1e2 / (constants.Boltzmann * temp)
    # plancks function
    Source = (C1 / (torch.exp(C2) - 1)) # in W m^2/cm^3/sr
    # for number density of air molec / m^3 and 1e2 for pressure values from hPa to Pa
    num_mole = press * 1e2 / (constants.Boltzmann * temp)
    kmTom = 1e3  # for dx integration
    # 1e4 for W cm/cm^2 to W cm/m^2 and S[ind, 0] in cm^2 / molec
    theta_scale = num_mole * 1e4 * S[ind,0] * kmTom

    A_scal = LineIntScal * Source * theta_scale

    A = A_lin * A_scal.T

    return A, 1

def torch_calcNonLin(tang_heights, dxs,  height_values, pressure_values, ind, temp_values, VMR_O3, wvnmbr, S, E,g_doub_prime,g_prime):
    '''careful that A_lin is just dx values
    maybe do A_lin_copy = np.copy(A_lin/2)
    A_lin_copy[:,-1] = A_lin_copy[:,-1] * 2
    if A_lin has been generated for linear data'''
    #print(dxs)
    # from : https://hitran.org/docs/definitions-and-units/
    # all calc in CGS
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]  # in cm^-1

    Q = g_doub_prime[ind, 0] * torch.exp(- HitrConst2 * E[ind, 0] / temp_values) + g_prime[ind, 0] * torch.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / temp_values)
    Q_ref = g_doub_prime[ind, 0] * torch.exp(- HitrConst2 * E[ind, 0] / 296) + g_prime[ind, 0] * torch.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / 296)
    LineIntScal = Q_ref / Q * torch.exp(- HitrConst2 * E[ind, 0] / temp_values) / torch.exp(
        - HitrConst2 * E[ind, 0] / 296) * (
                          1 - torch.exp(- HitrConst2 * wvnmbr[ind, 0] / temp_values)) / (
                          1 - torch.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))


    num_mole = 1 / constants.Boltzmann
    # 1e-4 cm^2/molec to m^2/molec
    theta = num_mole * VMR_O3 * S[ind,0] * 1e-4
    # 1e2 for pressure hPa to Pa and 1e5 for km to m
    ConcVal = - pressure_values * 1e2 * LineIntScal / temp_values * theta * 1e3
    #print( ConcVal)
    SpecNumMeas = len(tang_heights)
    SpecNumLayers = len(VMR_O3)

    afterTrans = torch.zeros((SpecNumMeas, SpecNumLayers))
    preTrans = torch.zeros((SpecNumMeas, SpecNumLayers))
    for i in range(0,SpecNumMeas):
        t = 0
        #print(np.array(height_values[0,t]))
        #print(np.array(tang_heights[i]))
        #print(np.array(tang_heights[i]) == np.array(height_values[0,t]))
        while height_values[t] <= tang_heights[i]:
            t += 1
        #print(dxs[i, t - 1:])
        flipDxs = torch.flip(dxs[i, t - 1:],dims =[0])
        #print(np.flip(np.array(dxs[i, t - 1:])) == np.array(torch.flip(dxs[i, t - 1:],dims =[0])))
        #print(flipDxs)
        flipVal = torch.flip(ConcVal[t - 1:],dims =[0])
        #print(np.flip(np.array(ConcVal[t - 1:])) == np.array(torch.flip(ConcVal[t - 1:],dims =[0])))

       
        currDxs = torch_gen_trap_rul(torch.cat((flipDxs, dxs[i, t - 1].reshape(-1)),dim = 0))
        #print(torch.cat((flipDxs, dxs[i, t - 1].reshape(-1)),dim = 0).sum())
        #print(torch.cat((flipVal.flatten() , ConcVal[t]),dim = 0))
        #print( ConcVal[t])
        #print(currDxs)
        ValPerLayAfter = torch.sum(torch.cat((flipVal.flatten() , ConcVal[t]),dim = 0) * currDxs)
        afterTrans[i, t - 1] = torch.exp(ValPerLayAfter)
        for j in range(t-1, SpecNumLayers-1):
            #print(len(dxs[i,j:]))
            currDxs = torch_gen_trap_rul(dxs[i,j:])
            #print(currDxs.sum())
            ValPerLayPre = torch.sum(ConcVal[j:].T  * currDxs)
            
            preTrans[i,j] = torch.exp(ValPerLayPre)

            if j >= t:
                currDxs = torch_gen_trap_rul(torch.cat((flipDxs, dxs[i, t - 1:j]),dim = 0))
                ValPerLayAfter = torch.sum(torch.cat((flipVal , ConcVal[t:j + 1]),dim = 0).flatten() * currDxs)
                afterTrans[i, j] = torch.exp(ValPerLayAfter)

        currDxs = torch_gen_trap_rul(torch.cat((flipDxs, dxs[i, t - 1:]),dim = 0))
        #print(currDxs.sum())
        ValPerLayAfter = torch.sum(torch.cat((torch.flip(ConcVal[t - 1:],dims =[0]), ConcVal[t:]),dim = 0).flatten() * currDxs)
        #print('------')
        #print(ValPerLayAfter)
        #print(currDxs)
        #print(torch.cat((torch.flip(ConcVal[t - 1:],dims =[0]), ConcVal[t:]),dim = 0).flatten())
        afterTrans[i, -1] = np.exp(ValPerLayAfter)
        preTrans[i, -1] = 1

    return preTrans + afterTrans

def torch_add_noise_Blokk(Ax,SNR):
    stdNoise = torch.max(Ax)/SNR
    return Ax + torch.normal(0,stdNoise, size= (len(Ax), 1)) , 1/stdNoise**2


def gen_data(ozon_val, hei_val, temp_val, pres_val, obs_height, r_earth, meas_angChosen, ind, SNR):


    A_lin_dx, tang_heights_lin, extraHeight = torch_gen_forward_map(meas_angChosen,hei_val,obs_height,r_earth)
    m = len(tang_heights_lin)
    A_lin = torch_gen_sing_map(A_lin_dx, tang_heights_lin, hei_val)
    
    tot_r = torch.zeros((m,1))
    #calculate total length
    for j in range(0, m):
        tot_r[j] =  torch.sqrt( ( hei_val[-1] + r_earth)**2 - (tang_heights_lin[j] +r_earth )**2)
    print('Distance through layers check: ' + str(torch.allclose( A_lin_dx.T.sum(0), tot_r[:,0])))
    files = '634f1dc4.par' 

    wvnmbr, S, F, g_air, g_self, E, n_air, g_doub_prime, g_prime = load_hitran_param(files)
    
    #load constants in si annd convert to cgs units by multiplying
    h = scy.constants.h #* 1e7#in J Hz^-1
    c_cgs = constants.c * 1e2# in m/s
    k_b = constants.Boltzmann #* 1e7#in J K^-1
    #T = temp_values[0:-1] #in K
    N_A = constants.Avogadro # in mol^-1
    R = constants.gas_constant
    mol_M = 48 #g/mol for Ozone

    #ind = 623 
    #pick wavenumber in cm^-1
    v_0 = wvnmbr[ind][0]#*1e2
    #wavelength
    lamba = 1/v_0
    f_0 = c_cgs*v_0
    
    
    print(f"Targeted frequency {v_0*c_cgs/1e9:.2f} in GHz")
    AParam = ind, wvnmbr, g_doub_prime, g_prime, E, S
    AO3, theta_scale_O3 = torch_composeAforO3(A_lin, temp_val, pres_val, *AParam)
    A = 2*AO3
    Ax = torch.matmul(A, ozon_val * theta_scale_O3)
    linNoiseFreeDat = Ax

    
    nonLinA = torch_calcNonLin(tang_heights_lin, A_lin_dx, hei_val, pres_val, ind, temp_val, ozon_val, wvnmbr, S, E,g_doub_prime,g_prime)
    
    
    OrgData = torch.matmul(AO3 * nonLinA,ozon_val * theta_scale_O3)
    
    nonLinY, gam0  = torch_add_noise_Blokk(OrgData,SNR)
    diff_linear_non_lin = torch.sqrt(torch.sum((Ax-OrgData)**2)/torch.sum(Ax**2))*100
    print(f'The difference between linear and non linear forward model is approx {diff_linear_non_lin:.1f}%' )
    return nonLinY, OrgData, tang_heights_lin, AParam



def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = 1#(5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def torch_f(ATy, y, B_inv_A_trans_y):
    return torch.matmul(y.T, y) - torch.matmul(ATy.T, B_inv_A_trans_y)

def torch_press(x, params: Tensor) -> Tensor:
    b = params[2]
    p0 = params[5]
    return torch.exp(-b * x  + torch.log(p0))

def torch_temp(height, params: Tensor) -> Tensor:
    a = torch.ones(height.shape[0])
    b = torch.ones(height.shape[0])

    b0 = params[4]
    a0 = params[9]
    a1 = params[7]
    a2 = params[10]
    a3 = params[11]
    a4 = params[13]
    a5 = params[15]
    a6 = params[17]
    h0 = params[3]
    h1 = params[8]
    h2 = params[6]
    h3 = params[12]
    h4 = params[14]
    h5 = params[16]

        
    a[height< h0] = a0
    a[h0 <= height] = a1
    a[h1 <= height] = a2
    a[h2 <= height] = a3
    a[h3 <= height] = a4
    a[h4 <= height] = a5
    a[h5 <= height] = a6
    #a[h6 <= x ] = 0

    b[height < h0] = b0
    b[h0 <= height] = b0 + h0 * a0
    b[h1 <= height] = b0 + (h1 - h0) * a1 + h0 * a0
    b[h2 <= height] = a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h3 <= height] = a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h4 <= height] = a4 * (h4 -h3) + a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    b[h5 <= height] = a5 * (h5 -h4) + a4 * (h4 -h3) + a3 * (h3-h2) + a2 * (h2-h1) + b0 + (h1 - h0) * a1 + h0 * a0
    #b[h6 <= x ] = a4 * (h6-h5) + a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0


    h = torch.ones(height.shape[0])
    h[height < h0] = 0
    h[h0 <= height] = h0
    h[h1 <= height] = h1
    h[h2 <= height] = h2
    h[h3 <= height] = h3
    h[h4 <= height] = h4
    h[h5 <= height] = h5
    #h[h6 <= x] = h6
    return a * (height - h) + b
    
def torch_marg_post(params: Tensor, h, RealMap, Alin, AParam, L, y, means, sigmas) -> Tensor:
    m, n = Alin.shape

    h1Mean = means[6]
    h1Sigm = sigmas[6]

    h2Mean = means[4]
    h2Sigm = sigmas[4]

    h3Mean = means[10]
    h3Sigm = sigmas[10]

    h4Mean = means[12]
    h4Sigm = sigmas[12]

    h5Mean = means[14]
    h5Sigm = sigmas[14]

    #h6Mean = means[6]
    #h6Sigm = sigmas[6]

    a0Mean = means[7]
    a0Sigm = sigmas[7]

    a1Mean = means[5]
    a1Sigm = sigmas[5]

    a2Mean = means[8]
    a2Sigm = sigmas[8]

    a3Mean = means[9]
    a3Sigm = sigmas[9]

    a4Mean = means[11]
    a4Sigm = sigmas[11]

    a5Mean = means[13]
    a5Sigm = sigmas[13]

    a6Mean = means[15]
    a6Sigm = sigmas[15]

    b0Mean = means[2]
    b0Sigm = sigmas[2]

    h0Mean = means[1]
    h0Sigm = sigmas[1]

    sigmaGrad1 = sigmas[0]
    bmean = means[0]
    sigmaP = sigmas[3]
    pmean = means[3]
    betaD = 1e-35
    betaG = 1e-35
    marg_post = torch.zeros(params.shape[0])
    for i in range(0, params.shape[0]):
        x = params[i,:]
        #lamb = x[1]
        delt = x[1]
        gam = x[0]
        h1 = x[8]
        h2 = x[6]
        h3 = x[12]
        h4 = x[14]
        h5 = x[16]
        a0 = x[9]
        a1 = x[7]
        a2 = x[10]
        a3 = x[11]
        a4 = x[13]
        a5 = x[15]
        a6 = x[17]
        b0 = x[4]
        h0 = x[3]
        b = x[2]
        p0 = x[5]

        lamb = delt/gam
        P = torch_press(h[:, 0], x).reshape((n, 1))
        T = torch_temp(h[:, 0], x).reshape((n, 1))

        CalcA, theatscale = torch_composeAforO3(Alin, T, P, *AParam)
        CurrA = RealMap @ CalcA
   

        Bp = CurrA.T @ CurrA + lamb * L
        LowTri = torch.linalg.cholesky(Bp)
     
        G = 2 * torch.log(torch.diag(LowTri)).sum()
        # G = g(CurrA, L, lamb)
        ATy = CurrA.T @ y
        B_inv_A_trans_y = torch.cholesky_solve(ATy, LowTri)
        F = torch_f(ATy, y, B_inv_A_trans_y)
        priors = (- ((h0 - h0Mean) / h0Sigm) ** 2 - ((h1 - h1Mean) / h1Sigm) ** 2 - ((h2 - h2Mean) / h2Sigm) ** 2 - (
                (h3 - h3Mean) / h3Sigm) ** 2 - ((h4 - h4Mean) / h4Sigm) ** 2
                  - ((h5 - h5Mean) / h5Sigm) ** 2 - ((a0 - a0Mean) / a0Sigm) ** 2
                  - ((a1 - a1Mean) / a1Sigm) ** 2 - ((a2 - a2Mean) / a2Sigm) ** 2
                  - ((a3 - a3Mean) / a3Sigm) ** 2 - ((a4 - a4Mean) / a4Sigm) ** 2
                  - ((a6 - a6Mean) / a6Sigm) ** 2 - ((a5 - a5Mean) / a5Sigm) ** 2
                  - ((b0 - b0Mean) / b0Sigm) ** 2
                  - ((pmean - p0) / sigmaP) ** 2 - ((bmean - b) / sigmaGrad1) ** 2)
        gamLamPrior = n/2 * torch.log(lamb) + (m/2 + 1) * torch.log(gam) -  ( betaD *  lamb * gam + betaG *gam)
        PrevMarg = - 0.5 * G - 0.5 * gam * F
        marg_post[i] = PrevMarg + 0.5 * priors + gamLamPrior -400
    #print('----')
    #print(marg_post.shape)
    #print(torch.max((marg_post)))
    #print(torch.min((marg_post)))
    return marg_post

def torch_RTO(A_lin, y, RealMap, cholesky_L, L, params, height_values, AParam):
    n = len(height_values)
    delt = params[1]
    gam = params[0]
    h1 = params[8]
    h2 = params[6]
    h3 = params[12]
    h4 = params[14]
    h5 = params[16]
    a0 = params[9]
    a1 = params[7]
    a2 = params[10]
    a3 = params[11]
    a4 = params[13]
    a5 = params[15]
    a6 = params[17]
    b0 = params[4]
    h0 = params[3]
    b = params[2]
    p0 = params[5]
    lamb = delt/gam
    P = torch_press(height_values[:, 0], params).reshape((n, 1))
    T = torch_temp(height_values[:, 0], params).reshape((n, 1))
    CalcA, theatscale = torch_composeAforO3(A_lin, T, P, *AParam)
    CurrA = RealMap @ CalcA
    ATy = CurrA.T @ y
    ATA = CurrA.T @ CurrA
    W = torch.normal(mean=torch.zeros(CurrA.shape[0]+CurrA.shape[1]), std=torch.ones(CurrA.shape[0]+CurrA.shape[1]))
    #W = np.random.normal(loc=0.0, scale=1, size=len(CurrA))
    v_1 = np.sqrt(gam) * CurrA.T @ W[:CurrA.shape[0]]

    #mvn = MultivariateNormal(loc=torch.zeros(CurrA.shape[1]), covariance_matrix=L)
    #W2 = mvn.sample()
    #W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    v_2 = np.sqrt(delt) * (cholesky_L @ W[CurrA.shape[0]:])

    SetB = gam * ATA + delt * L
    RandX = (gam * ATy + v_1.reshape((n, 1)) + v_2.reshape((n, 1)))
    #print(RandX.shape)
    LowTri = torch.linalg.cholesky(SetB)
    UpTri = LowTri.T
    
    XSampl = torch.cholesky_solve(RandX, LowTri)



    return XSampl[:,0] , P[:,0], T[:,0]
