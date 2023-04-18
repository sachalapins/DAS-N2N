import numpy as np
from scipy.signal import convolve
from scipy.special import factorial

def ray_trace_hom_vti(s,r,C,rho):
    """
    Return ray attributes for given source and reciever locations and VTI stiffness tensor.
    Based on equations from Chapman (2004): Fundamentals of seismic wave propagation, 
    and Leaney (2014): Microseismic source inversion in anisotropic media.
    
    Arguments:
    s[3] -- source coordinates 
    r[3] -- receiver coordinate
    C[3,3] -- VTI stiffness tensor in Voigt notation
    rho -- density
    
    Output:
    TqP  -- qP travel time 
    TqSv -- qSv travel time 
    TSh  -- Sh travel time 
    vqP  -- qP phase velocity
    vqSv -- qSv phase velocity
    vSh  -- qSh phase velocity
    VqP  -- qP group velocity
    VqSv -- qSv group velocity
    VSh  -- qSh group velocity
    SqP  -- qP spreading factor
    SqSv -- qSv spreading factor
    SSh  -- qSh spreading factor
    gqP  -- qP polarization vector
    gqSv -- qSv polarization vector
    gSh  -- qSh polarization vector
    qPp  -- qP slowness vector
    qSvp -- qSv slowness vector
    Shp  -- qSh slowness vector
    
    """


    s=np.array(s)
    r=np.array(r)

    # Normalize stiffness by density
    A = C/rho


    a = A[0,2] + A[3,3]
    Ac = A[0,0]*A[2,2]+A[3,3]**2-a**2

    dh = (np.sum((r[0:2]-s[0:2])**2))**0.5
    dv = np.abs(r[2] - s[2])

    # HORIZONTAL WAVE SPEEDS (Leaney PhD thesis, pp. 47, Section 3.6)
    Vpmax = A[0,0]**0.5
    Vsvmax = A[3,3]**0.5
    Vshmax = A[5,5]**0.5

    # VERTICAL WAVE SPEEDS
    Vpv = A[2,2]**0.5
    Vsvv = A[3,3]**0.5
    Vshv = A[3,3]**0.5

    # HORIZONTAL SLOWNESS VECTORS FOR VERTICAL PROPAGATION
    Pp10 = 0
    Svp10 = 0
    Shp10 = 0
    # HORIZONTAL SLOWNESS FOR HORIZONTAL PROPAGATION (Leaney PhD thesis, pp. 47, Section 3.6)
    Pp1f = 1/Vpmax
    Svp1f = 1/Vsvmax
    Shp1f = 1/Vshmax
    # HORIZONTAL SLOWNESS FOR INTERMEDIATE PROPAGATION
    Pp1m = Pp1f/2
    Svp1m = Svp1f/2
    Shp1m = Shp1f/2

    # MAXIMUM NUMBER OF ITERATIONS
    maxit = 5000




    # UPDATE SLOWNESS VECTOR FOR qP WAVES
    if dv == 0:
        Pp1 = Pp10
        BP = A[2,2] + A[3,3] - Ac*Pp1**2
        Pp3 = ( (BP - ( BP**2 - 4*A[2,2]*A[3,3]*(A[0,0]*Pp1**2 - 1)* (A[3,3]*Pp1**2 - 1) )**0.5)/(2*A[2,2]*A[3,3]) )**0.5
        BP = None
    elif dh == 0:
        Pp1 = Pp1f
        Pp3 = 0
    else:
        it = 0
        while True:

            # INCREASE ITERATION COUNTER
            it = it + 1

            # CONSTANT FACTOR (Chapman 2004, eq. 5.7.33, pp. 187)
            BPp10 = A[2,2] + A[3,3] - Ac*Pp10**2
            BPp1f = A[2,2] + A[3,3] - Ac*Pp1f**2
            BPp1m = A[2,2] + A[3,3] - Ac*Pp1m**2

            # VERTICAL SLOWNESS COMPONENT (Chapman 2004, eq. 5.7.32, pp. 187)
            Pp30 = ( (BPp10 - ( BPp10**2 - 4*A[2,2]*A[3,3]*(A[0,0]*Pp10**2 - 1)*(A[3,3]*Pp10**2 - 1) )**0.5)/(2*A[2,2]*A[3,3]) )**0.5
            Pp3f = ( (BPp1f - ( BPp1f**2 - 4*A[2,2]*A[3,3]*(A[0,0]*Pp1f**2 - 1)*(A[3,3]*Pp1f**2 - 1) )**0.5)/(2*A[2,2]*A[3,3]) )**0.5
            Pp3m = ( (BPp1m - ( BPp1m**2 - 4*A[2,2]*A[3,3]*(A[0,0]*Pp1m**2 - 1)*(A[3,3]*Pp1m**2 - 1) )**0.5)/(2*A[2,2]*A[3,3]) )**0.5

            # HORIZONTAL RANGES (Chapman 2004, eq. 5.7.39, pp. 188)
            XPp10 = Pp10*( 2*A[0,0]*A[3,3]*Pp10**2 + Ac*Pp30**2 - A[0,0] - A[3,3] )*dv/(Pp30*( Ac*Pp10**2 + 2*A[2,2]*A[3,3]*Pp30**2 - A[2,2] - A[3,3] ))
            XPp1f = Pp1f*( 2*A[0,0]*A[3,3]*Pp1f**2 + Ac*Pp3f**2 - A[0,0] - A[3,3] )*dv/(Pp3f*( Ac*Pp1f**2 + 2*A[2,2]*A[3,3]*Pp3f**2 - A[2,2] - A[3,3] ))
            XPp1m = Pp1m*( 2*A[0,0]*A[3,3]*Pp1m**2 + Ac*Pp3m**2 - A[0,0] - A[3,3] )*dv/(Pp3m*( Ac*Pp1m**2 + 2*A[2,2]*A[3,3]*Pp3m**2 - A[2,2] - A[3,3] ))

            # CONTINUE ITERATION IN BETWEEN THE SLOWNESSES THAT LAND CLOSER TO THE
            # DESIRED POSITION
            dx0 = abs(XPp10 - dh)
            dxm = abs(XPp1m - dh)
            dxf = abs(XPp1f - dh)
            closest=np.argmin([dx0,dxm,dxf])
            if closest == 0:
                X = XPp10
                p1 = Pp10
                p3 = Pp30
            elif closest == 1:
                X = XPp1m
                p1 = Pp1m
                p3 = Pp3m
            else:
                X = XPp1f
                p1 = Pp1f
                p3 = Pp3f

            # CHECK FOR CONVERGENCE
            if abs(X - dh) < 1e-5 or it == maxit:
                Pp1 = p1
                Pp3 = p3
                if it == maxit:
                    print('WARNING: non-convergence, max #iterations reached for qP-waves')
                break
            else:
                # UPDATE HORIZONTAL SLOWNESS FOR NEXT ITERATION
                if XPp1m - dh < 0:
                    Pp10 = Pp1m
                else:
                    Pp1f = Pp1m
                Pp1m = (Pp10 + Pp1f)/2


    # UPDATE SLOWNESS VECTOR FOR qSv WAVES
    if dv == 0:
        Svp1 = Svp10
        BS = A[2,2] + A[3,3] - Ac*Svp1**2
        Svp3 = ( (BS + ( BS**2 - 4*A[2,2]*A[3,3]*(A[0,0]*Svp1**2 - 1)*(A[3,3]*Svp1**2 - 1) )**0.5)/(2*A[2,2]*A[3,3]) )**0.5
        BS = None
    elif dh == 0:
        Svp1 = Svp1f
        Svp3 = 0
    else:
        it = 0
        while True:

            # INCREASE ITERATION COUNTER
            it = it + 1

            # CONSTANT FACTOR (Chapman 2004, eq. 5.7.33, pp. 187)
            BSvp10 = A[2,2] + A[3,3] - Ac*Svp10**2
            BSvp1f = A[2,2] + A[3,3] - Ac*Svp1f**2
            BSvp1m = A[2,2] + A[3,3] - Ac*Svp1m**2

            # VERTICAL SLOWNESS COMPONENT (Chapman 2004, eq. 5.7.32, pp. 187)
            Svp30 = ( (BSvp10 + ( BSvp10**2 - 4*A[2,2]*A[3,3]*(A[0,0]*Svp10**2 - 1)*(A[3,3]*Svp10**2 - 1) )**0.5)/(2*A[2,2]*A[3,3]) )**0.5
            Svp3f = ( (BSvp1f + ( BSvp1f**2 - 4*A[2,2]*A[3,3]*(A[0,0]*Svp1f**2 - 1)*(A[3,3]*Svp1f**2 - 1) )**0.5)/(2*A[2,2]*A[3,3]) )**0.5
            Svp3m = ( (BSvp1m + ( BSvp1m**2 - 4*A[2,2]*A[3,3]*(A[0,0]*Svp1m**2 - 1)*(A[3,3]*Svp1m**2 - 1) )**0.5)/(2*A[2,2]*A[3,3]) )**0.5

            # HORIZONTAL RANGES (Chapman 2004, eq. 5.7.39, pp. 188)
            XSvp10 = Svp10*( 2*A[0,0]*A[3,3]*Svp10**2 + Ac*Svp30**2 - A[0,0] - A[3,3] )*dv/(Svp30*( Ac*Svp10**2 + 2*A[2,2]*A[3,3]*Svp30**2 - A[2,2] - A[3,3] ))
            XSvp1f = Svp1f*( 2*A[0,0]*A[3,3]*Svp1f**2 + Ac*Svp3f**2 - A[0,0] - A[3,3] )*dv/(Svp3f*( Ac*Svp1f**2 + 2*A[2,2]*A[3,3]*Svp3f**2 - A[2,2] - A[3,3] ))
            XSvp1m = Svp1m*( 2*A[0,0]*A[3,3]*Svp1m**2 + Ac*Svp3m**2 - A[0,0] - A[3,3] )*dv/(Svp3m*( Ac*Svp1m**2 + 2*A[2,2]*A[3,3]*Svp3m**2 - A[2,2] - A[3,3] ))

            # CONTINUE ITERATION IN BETWEEN THE SLOWNESSES THAT LAND CLOSER TO THE
            # DESIRED POSITION
            dx0 = abs(XSvp10 - dh)
            dxm = abs(XSvp1m - dh)
            dxf = abs(XSvp1f - dh)
            closest=np.argmin([dx0,dxm,dxf])
            if closest == 0:
                X = XSvp10
                p1 = Svp10
                p3 = Svp30
            elif closest == 1:
                X = XSvp1m
                p1 = Svp1m
                p3 = Svp3m
            else:
                X = XSvp1f
                p1 = Svp1f
                p3 = Svp3f


            # CHECK FOR CONVERGENCE
            if abs(X - dh) < 1e-5 or it == maxit:
                Svp1 = p1
                Svp3 = p3
                if it == maxit:
                    print('WARNING: non-convergence, max #iterations reached for qSv-waves')
                break
            else:
                # UPDATE HORIZONTAL SLOWNESS FOR NEXT ITERATION
                if XSvp1m - dh < 0:
                    Svp10 = Svp1m
                else:
                    Svp1f = Svp1m
                Svp1m = (Svp10 + Svp1f)/2


    # SLOWNESS VECTOR FOR Sh WAVES
    if dv == 0:
        Shp1 = Shp10
        Shp3 = ( 1/A[3,3] - Shp1**2*A[5,5]/A[3,3] )**0.5
    elif dh == 0:
        Shp1 = Shp1f
        Shp3 = 0
    else:
        it = 0
        while True:

            # INCREASE ITERATION COUNTER
            it = it + 1

            # VERTICAL SLOWNESS COMPONENT (Chapman 2004, eqs. 5.7.24, pp. 186)
            Shp30 = ( 1/A[3,3] - Shp10**2*A[5,5]/A[3,3] )**0.5
            Shp3f = ( 1/A[3,3] - Shp1f**2*A[5,5]/A[3,3] )**0.5
            Shp3m = ( 1/A[3,3] - Shp1m**2*A[5,5]/A[3,3] )**0.5

            # HORIZONTAL RANGES (Chapman 2004, eqs. 5.7.27, pp. 186)
            XShp10 = A[5,5]*Shp10*dv/(A[3,3]*Shp30)
            XShp1f = A[5,5]*Shp1f*dv/(A[3,3]*Shp3f)
            XShp1m = A[5,5]*Shp1m*dv/(A[3,3]*Shp3m)

            # CONTINUE ITERATION IN BETWEEN THE SLOWNESSES THAT LAND CLOSER TO THE
            # DESIRED POSITION
            dx0 = abs(XShp10 - dh)
            dxm = abs(XShp1m - dh)
            dxf = abs(XShp1f - dh)
            closest=np.argmin([dx0,dxm,dxf])
            if closest == 0:
                X = XShp10
                p1 = Shp10
                p3 = Shp30
            elif closest == 1:
                X = XShp1m
                p1 = Shp1m
                p3 = Shp3m
            else:
                X = XShp1f
                p1 = Shp1f
                p3 = Shp3f

            # CHECK FOR CONVERGENCE
            if abs(X - dh) < 1e-5 or it == maxit:
                Shp1 = p1
                Shp3 = p3
                if it == maxit:
                    print('WARNING: non-convergence, max #iterations reached for Sh-waves')
                break
            else:
                # UPDATE HORIZONTAL SLOWNESS FOR NEXT ITERATION
                if XShp1m - dh < 0:
                    Shp10 = Shp1m
                else:
                    Shp1f = Shp1m
                Shp1m = (Shp10 + Shp1f)/2



    ###########################################################################
    # TRAVEL TIMES (Chapman 2004, eqs. 5.7.28, 5.7.38, 5.7.40, pp. 187)
    DqP = (A[0,0] + A[3,3])*Pp1**2 + (A[2,2] + A[3,3])*Pp3**2 - 2
    DqSv = (A[0,0] + A[3,3])*Svp1**2 + (A[2,2] + A[3,3])*Svp3**2 - 2
    TqP = dv*DqP/( Pp3*(Ac*Pp1**2 + 2*A[2,2]*A[4,4]*Pp3**2 - A[2,2] - A[3,3]) )
    TqSv = dv*DqSv/( Svp3*(Ac*Svp1**2 + 2*A[2,2]*A[4,4]*Svp3**2 - A[2,2] - A[3,3]) )
    TSh = dv/(A[3,3]*Shp3)
    if dv == 0:
        TqP = dh/Vpmax
        TqSv = dh/Vsvmax
        TSh = dh/Vshmax

    if dh == 0:
        TqP = dv/Vpv
        TqSv = dv/Vsvv
        TSh = dv/Vshv

    ###########################################################################

    ###########################################################################
    # PHASE VELOCITIES (Leaney PhD thesis eq. 3.31, pp. 48)
    vqP = 1/(Pp1**2 + Pp3**2)**0.5
    vqSv = 1/(Svp1**2 + Svp3**2)**0.5
    vSh = 1/(Shp1**2 + Shp3**2)**0.5
    if dv == 0:
        vqP = Vpmax
        vqSv = Vsvmax
        vSh = Vshmax
    if dh == 0:
        vqP = Vpv
        vqSv = Vsvv
        vSh = Vshv

    ###########################################################################

    ###########################################################################
    # GROUP VELOCITIES (Chapman 2004, eqs. 5.7.36, 5.7.37, pp. 187)
    # Leaney PhD thesis eq. 3.32, pp. 49
    VPp1 = Pp1*( 2*A[0,0]*A[3,3]*Pp1**2 + Ac*Pp3**2 - A[0,0] - A[3,3] )/DqP
    VPp3 = Pp3*( Ac*Pp1**2 + 2*A[2,2]*A[3,3]*Pp3**2 - A[2,2] - A[3,3] )/DqP
    VqP = (VPp1**2 + VPp3**2)**0.5
    VSvp1 = Svp1*( 2*A[0,0]*A[3,3]*Svp1**2 + Ac*Svp3**2 - A[0,0] - A[3,3] )/DqSv
    VSvp3 = Svp3*( Ac*Svp1**2 + 2*A[2,2]*A[3,3]*Svp3**2 - A[2,2] - A[3,3] )/DqSv
    VqSv = (VSvp1**2 + VSvp3**2)**0.5
    VShp1 = A[5,5]*Shp1
    VShp3 = A[3,3]*Shp3
    VSh = (VShp1**2 + VShp3**2)**0.5
    if dv == 0:
        VqP = Vpmax
        VqSv = Vsvmax
        VSh = Vshmax
    if dh == 0:
        VqP = Vpv
        VqSv = Vsvv
        VSh = Vshv
    ###########################################################################

    ###########################################################################
    # PHASE ANGLES
    thetaP = np.arcsin(Pp1*vqP)
    thetaSv = np.arcsin(Svp1*vqSv)
    thetaSh = np.arcsin(Shp1*vSh)
    ###########################################################################

    ###########################################################################
    # GROUP ANGLE (Thomsen 1986, Fig. 1)
    phi = np.arctan(dh/dv)
    ###########################################################################

    ###########################################################################
    # SPREADING FACTORS (Chapman 2004, eqs. 5.7.29, 5.7.41, pp. 187-188)
    SqP = (dh/Pp1)*(1 + Pp1*dh/(Pp3*dv)) + (4/(Pp3*dv))*(A[0,0]*A[3,3]*Pp1**2*dv**2 - Ac*Pp1*Pp3*dh*dv + A[2,2]*A[3,3]*Pp3**2*dh**2)/(Ac*Pp1**2 + 2*A[2,2]*A[3,3]*Pp3**2 - A[2,2] - A[3,3])
    SqSv = (dh/Svp1)*(1 + Svp1*dh/(Svp3*dv)) + (4/(Svp3*dv))*(A[0,0]*A[3,3]*Svp1**2*dv**2 - Ac*Svp1*Svp3*dh*dv + A[2,2]*A[3,3]*Svp3**2*dh**2)/(Ac*Svp1**2 + 2*A[2,2]*A[3,3]*Svp3**2 - A[2,2] - A[3,3])
    SSh = A[5,5]*dv/(A[3,3]**2*Shp3**3)
    # SqP = abs(np.cos(phi))*abs(np.cos(thetaP))*abs(SqP*dh/Pp1)
    # SqSv = abs(np.cos(phi))*abs(np.cos(thetaSv))*abs(SqSv*dh/Svp1)
    # SSh = abs(np.cos(phi))*abs(np.cos(thetaSh))*abs(SSh*dh/Shp1)
    # Spreading factors from Leaney PhD thesis eqs 3.34, pp 49
    SqP = abs(np.cos(phi))**2*abs(SqP*dh/Pp1)
    SqSv = abs(np.cos(phi))**2*abs(SqSv*dh/Svp1)
    SSh = abs(np.cos(phi))**2*abs(SSh*dh/Shp1)
    if dv/dh < 0.02: # this threshold was set arbitrarily to account for subhorizontal rays
        # EFFECTIVE MEDIUM SPREADING (Leaney PhD thesis section 3.8.3, pp. 50-51)
        SqP = (dh*vqP)**2
        SqSv = (dh*vqSv)**2
        SSh = (dh*vSh)**2
    if dh/dv < 0.02: # this threshold was set arbitrarily to account for subvertical rays
        # EFFECTIVE MEDIUM SPREADING (Leaney PhD thesis section 3.8.3, pp. 50-51)
        SqP = (dv*vqP)**2
        SqSv = (dv*vqSv)**2
        SSh = (dv*vSh)**2

    ###########################################################################

    ###########################################################################
    # POLARIZATION VECTORS
    tmp = (Pp1**2 + Pp3**2)**0.5
    Pp1 = Pp1/tmp
    Pp3 = Pp3/tmp
    gqP=np.zeros([3])
    gqP[0] = 0
    gqP[1] = 2*a*Pp1*Pp3
    gqP[2] = (A[2,2] - A[3,3])*Pp3**2 - (A[0,0] - A[3,3])*Pp1**2 + ( ( (A[2,2] - A[3,3])*Pp3**2 - (A[0,0] - A[3,3])*Pp1**2 )**2 + 4*Pp1**2*Pp3**2*a**2 )**0.5
    gqP = gqP/np.linalg.norm(gqP)
    tmp = (Svp1**2 + Svp3**2)**0.5
    Svp1 = Svp1/tmp
    Svp3 = Svp3/tmp
    gqSv=np.zeros([3])
    gqSv[0] = 0
    gqSv[1] = 2*a*Svp1*Svp3
    gqSv[2] = (A[2,2] - A[3,3])*Svp3**2 - (A[0,0] - A[3,3])*Svp1**2 -( ( (A[2,2] - A[3,3])*Svp3**2 - (A[0,0] - A[3,3])*Svp1**2 )**2 + 4*Svp1**2*Svp3**2*a**2 )**0.5
    gqSv = gqSv/np.linalg.norm(gqSv)
    gSh = np.array([1.,0,0])
    if dv == 0:
        gqP = np.array([0,1.,0])
        gqSv = np.array([0,0,-1.])
    if dh == 0:
        gqP = np.array([0,0,1.])
        gqSv = np.sign(r[2] - s[2])*np.array([0,1.,0])
    ###########################################################################

    ###########################################################################
    # SLOWNESS VECTORS
    qPp = np.array([0,Pp1,Pp3])
    qPp = qPp/np.linalg.norm(qPp)
    qSvp = np.array([0,Svp1,Svp3])
    qSvp = qSvp/np.linalg.norm(qSvp)
    Shp = np.array([0,Shp1,Shp3])
    Shp = Shp/np.linalg.norm(Shp)
    if dv == 0:
        qPp =  np.array([0,1.,0])
        qSvp = np.array([0,1.,0])
        Shp =  np.array([0,1.,0])
    if dh == 0:
        qPp =  np.array([0,0,1.])
        qSvp = np.array([0,0,1.])
        Shp =  np.array([0,0,1.])
    ###########################################################################

    # ROTATE POLARIZATION AND SLOWNESS VECTORS TO ORIGINAL COORDINATE SYSTEM
    az = np.arctan2(r[0]-s[0],r[1]-s[1])

    # ROTATION MATRIX IN THE HORIZONTAL PLANE
    R = np.array([[np.cos(az), np.sin(az)],
                 [-np.sin(az), np.cos(az)]])

    # ROTATIONS
    gqP[0:2] = np.dot(R,gqP[0:2].T)
    gqSv[0:2] = np.dot(R,gqSv[0:2].T)
    gSh[0:2] = np.dot(R,gSh[0:2].T)
    qPp[0:2] = np.dot(R,qPp[0:2].T)
    qSvp[0:2] = np.dot(R,qSvp[0:2].T)
    Shp[0:2] = np.dot(R,Shp[0:2].T)

    # CORRECT POLARIZATIONS BY SENSE OF PROPAGATION IN THE VERTICAL DIRECTION
    if r[2] - s[2] < 0:
        gqP[2] = -gqP[2]
        gqSv[2] = -gqSv[2]
        qPp[2] = -qPp[2]
        qSvp[2] = -qSvp[2]
        Shp[2] = -Shp[2]


    return TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp



def MTradiation(M33,polar,slow,grpvel,rho):
    """
    give mt, a slowness vector of a phase, 
    and a source polarization return radiation amplitude
    """
    # Normalize polarization by sqrt(2*rho*V)
    polarnrm=polar/np.sqrt(2*rho*grpvel)
    # compute (normalized) ray-deformation tensor
    raystrn=np.einsum('i,j->ij',polarnrm,slow)
    # convert to symmetric strain tensor
    raystrn=0.5*(raystrn+raystrn.T)
    # compute normalized source radiation amplitude
    srcamp= np.einsum('ij,ij',M33,raystrn)
    return srcamp


def ScalarPropagation(spreading):
    """
    Compute scale factor to account for propagation effects. 
    Currently only includes geometric spreading, 
    but can be expanded to include R/T coefficients
    """

    return 1/(2*np.pi*np.sqrt(spreading))

def RecDispl(polar,grpvel,rho):
    """
    Compute receiver displacements given polarization vector,
    group velocity vector and density at receiver
    """
    return polar/np.sqrt(2*rho*grpvel)

def RecStrain(polar,slow,grpvel,rho):
    """
    Compute receiver strain given polarization vector, 
    slowness vector, group velocity vector and density at receiver
    """
    # Normalize polarization by sqrt(2*rho*V)
    polarnrm=polar/np.sqrt(2*rho*grpvel)
    # compute (normalized) ray-strain tensor
    recstrain=np.einsum('i,j->ij',polarnrm,slow)
    # Added minus sign
    recstrain=-0.5*(recstrain+recstrain.T)
    return recstrain[np.triu_indices(3)]







def radpattern(src,x,C,rho,M33):
    if len(x.shape) > 1:
        Prad=np.empty(x.shape[0])
        Svrad=np.empty(x.shape[0])
        Shrad=np.empty(x.shape[0])

        for i in range(x.shape[0]):
            Prad[i], Svrad[i], Shrad[i] = radpattern(src,x[i,:],C,rho,M33)

    else:

        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh

        Prad=MTradiation(M33,gqP,slowP,VqP,rho)
        Svrad=MTradiation(M33,gqSv,slowSv,VqSv,rho)
        Shrad=MTradiation(M33,gSh,slowSh,VSh,rho)


    return Prad, Svrad, Shrad


def strainamps(src,x,C,rho,M33):

    if len(x.shape) > 1:
        Pstrains=np.empty((x.shape[0],6))
        Svstrains=np.empty((x.shape[0],6))
        Shstrains=np.empty((x.shape[0],6))

        for i in range(x.shape[0]):
            Pstrains[i,:], Svstrains[i,:], Shstrains[i,:] = strainamps(src,x[i,:],C,rho,M33)

    else:

        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh

        Prad=MTradiation(M33,gqP,slowP,VqP,rho)
        Svrad=MTradiation(M33,gqSv,slowSv,VqSv,rho)
        Shrad=MTradiation(M33,gSh,slowSh,VSh,rho)

        # Prad=1
        # Svrad=1
        # Shrad=1

        Pprop=ScalarPropagation(SqP)
        Svprop=ScalarPropagation(SqSv)
        Shprop=ScalarPropagation(SSh)

        # Pprop=1
        # Svprop=1
        # Shprop=1

        Pstrain=RecStrain(gqP,slowP,VqP,rho)
        Svstrain=RecStrain(gqSv,slowSv,VqSv,rho)
        Shstrain=RecStrain(gSh,slowSh,VSh,rho)


        Pstrains = Pstrain  * Pprop  * Prad
        Svstrains = Svstrain * Svprop * Svrad
        Shstrains = Shstrain * Shprop * Shrad

    return Pstrains, Svstrains, Shstrains
    
    
def strainamps_uniform(src,x,C,rho):

    if len(x.shape) > 1:
        Pstrains=np.empty((x.shape[0],6))
        Svstrains=np.empty((x.shape[0],6))
        Shstrains=np.empty((x.shape[0],6))

        for i in range(x.shape[0]):
            Pstrains[i,:], Svstrains[i,:], Shstrains[i,:] = strainamps_uniform(src,x[i,:],C,rho)

    else:

        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh



        Prad=1
        Svrad=1
        Shrad=1

        Pprop=ScalarPropagation(SqP)
        Svprop=ScalarPropagation(SqSv)
        Shprop=ScalarPropagation(SSh)

        # Pprop=1
        # Svprop=1
        # Shprop=1

        Pstrain=RecStrain(gqP,slowP,VqP,rho)
        Svstrain=RecStrain(gqSv,slowSv,VqSv,rho)
        Shstrain=RecStrain(gSh,slowSh,VSh,rho)


        Pstrains = Pstrain  * Pprop  * Prad
        Svstrains = Svstrain * Svprop * Svrad
        Shstrains = Shstrain * Shprop * Shrad

    return Pstrains, Svstrains, Shstrains




def slownesses(src,x,C,rho,M33):

    if len(x.shape) > 1:
        slowP=np.empty((x.shape[0],3))
        slowSv=np.empty((x.shape[0],3))
        slowSh=np.empty((x.shape[0],3))

        for i in range(x.shape[0]):
            slowP[i,:], slowSv[i,:], slowSh[i,:] = slownesses(src,x[i,:],C,rho,M33)

    else:

        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh



    return slowP, slowSv, slowSh


def slownesstimes(src,x,C,rho,M33):

    if len(x.shape) > 1:
        slowP=np.empty((x.shape[0],3))
        slowSv=np.empty((x.shape[0],3))
        slowSh=np.empty((x.shape[0],3))
        TqP=np.empty((x.shape[0]))
        TqSv=np.empty((x.shape[0]))
        TSh=np.empty((x.shape[0]))
        

        for i in range(x.shape[0]):
            slowP[i,:], slowSv[i,:], slowSh[i,:], TqP[i], TqSv[i], TSh[i]= slownesstimes(src,x[i,:],C,rho,M33)

    else:

        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh



    return slowP, slowSv, slowSh, TqP, TqSv, TSh






def displamps(src,x,C,rho,M33):

    if len(x.shape) > 1:
        Pdispls=np.empty((x.shape[0],3))
        Svdispls=np.empty((x.shape[0],3))
        Shdispls=np.empty((x.shape[0],3))

        for i in range(x.shape[0]):
            Pdispls[i,:], Svdispls[i,:], Shdispls[i,:] = displamps(src,x[i,:],C,rho,M33)

    else:

        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh

        Prad=MTradiation(M33,gqP,slowP,VqP,rho)
        Svrad=MTradiation(M33,gqSv,slowSv,VqSv,rho)
        Shrad=MTradiation(M33,gSh,slowSh,VSh,rho)

        # Prad=1
        # Svrad=1
        # Shrad=1

        Pprop=ScalarPropagation(SqP)
        Svprop=ScalarPropagation(SqSv)
        Shprop=ScalarPropagation(SSh)

        # Pprop=1
        # Svprop=1
        # Shprop=1

        Pdispl=RecDispl(gqP,VqP,rho)
        Svdispl=RecDispl(gqSv,VqSv,rho)
        Shdispl=RecDispl(gSh,VSh,rho)


        Pdispls = Pdispl  * Pprop  * Prad
        Svdispls = Svdispl * Svprop * Svrad
        Shdispls = Shdispl * Shprop * Shrad

    return Pdispls, Svdispls, Shdispls






def StrainBackprop(rec,x,C,rho,nsam,dt,signal,M33):
    """rec: list of receiver locations"""
    strainwaves=np.zeros((x.shape[0],nsam,6))
    for i in range(x.shape[0]):
        for j in range(rec.shape[0]):
            strainwaves[i,:,:]+=ForwAdjOpStrain(rec[j,:],x[i,:],C,rho,nsam,dt,signal[j,:],M33)

    return strainwaves




def ForwardModelStrain(src,x,C,rho,nsam,dt,w,M33,t0=0):
    """docstring for ForwardModelStrain"""


    if len(x.shape) > 1:
        strainwaves=np.empty((x.shape[0],nsam,6))

        for i in range(x.shape[0]):
            strainwaves[i,:,:]=ForwardModelStrain(src,x[i,:],C,rho,nsam,dt,w,M33,t0=t0)

    else:


        # write as if src and x are single points

        strains=np.zeros([nsam,6])
        # displs=np.zeros([nsam,3])

        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh

        Prad=MTradiation(M33,gqP,slowP,VqP,rho)
        Svrad=MTradiation(M33,gqSv,slowSv,VqSv,rho)
        Shrad=MTradiation(M33,gSh,slowSh,VSh,rho)

        Pprop=ScalarPropagation(SqP)
        Svprop=ScalarPropagation(SqSv)
        Shprop=ScalarPropagation(SSh)

        Pstrain=RecStrain(gqP,slowP,VqP,rho)
        Svstrain=RecStrain(gqSv,slowSv,VqSv,rho)
        Shstrain=RecStrain(gSh,slowSh,VSh,rho)

        # Pstrain=RecStrain(gqP,qPp,VqP,rho)
        # Svstrain=RecStrain(gqSv,qSvp,VqSv,rho)
        # Shstrain=RecStrain(gSh,Shp,VSh,rho)

        # Pdispl=RecDispl(gqP,VqP,rho)
        # Svdispl=RecDispl(gqSv,VqSv,rho)
        # Shdispl=RecDispl(gSh,VSh,rho)

        P = int(round(TqP/dt))
        Sv = int(round(TqSv/dt))
        Sh = int(round(TSh/dt))
        if P<nsam:
            strains[P,:]  += Pstrain  * Pprop  * Prad
        if Sv<nsam:
            strains[Sv,:] += Svstrain * Svprop * Svrad
        if Sh<nsam:
            strains[Sh,:] += Shstrain * Shprop * Shrad

        # Need to find the derivative of the displacement wavelet
        gradw=np.gradient(w,dt)
        
        

        strt = int(round(t0/dt))
        strainwaves=np.apply_along_axis(convolve,0,strains,gradw,mode='full')[strt:nsam+strt]
        
        

        strainwaves=strainwaves[:nsam]

        # displs[P,:]  += Pdispl  * Pprop  * Prad
        # displs[Sv,:] += Svdispl * Svprop * Svrad
        # displs[Sh,:] += Shdispl * Shprop * Shrad


    return strainwaves



def radiation(src,x,C,rho,M33):
    """docstring for ForwAdjOp"""


    if len(x.shape) > 1:
        rad=np.empty((x.shape[0],3))

        for i in range(x.shape[0]):
            rad[i,:]=radiation(src,x[i,:],C,rho,M33)

    else:



        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh

        Prad=MTradiation(M33,gqP,slowP,VqP,rho)
        Svrad=MTradiation(M33,gqSv,slowSv,VqSv,rho)
        Shrad=MTradiation(M33,gSh,slowSh,VSh,rho)

        rad=np.zeros(3)
        rad[0]=Prad
        rad[1]=Svrad
        rad[2]=Shrad


    return rad






def ForwardModelDispl(src,x,C,rho,nsam,dt,w,M33,t0=0):
    """docstring for ForwardModelDispl"""


    if len(x.shape) > 1:
        displwaves=np.empty((x.shape[0],nsam,3))

        for i in range(x.shape[0]):
            displwaves[i,:,:]=ForwardModelDispl(src,x[i,:],C,rho,nsam,dt,w,M33,t0=t0)

    else:


        # write as if src and x are single points

        # strains=np.zeros([nsam,6])
        displs=np.zeros([nsam,3])

        [TqP,TqSv,TSh,vqP,vqSv,vSh,VqP,VqSv,VSh,SqP,SqSv,SSh,gqP,gqSv,gSh,qPp,qSvp,Shp] = ray_trace_hom_vti(src,x,C,rho)

        slowP=qPp/vqP
        slowSv=qSvp/vqSv
        slowSh=Shp/vSh

        Prad=MTradiation(M33,gqP,slowP,VqP,rho)
        Svrad=MTradiation(M33,gqSv,slowSv,VqSv,rho)
        Shrad=MTradiation(M33,gSh,slowSh,VSh,rho)

        Pprop=ScalarPropagation(SqP)
        Svprop=ScalarPropagation(SqSv)
        Shprop=ScalarPropagation(SSh)

        # Pstrain=RecStrain(gqP,slowP,VqP,rho)
        # Svstrain=RecStrain(gqSv,slowSv,VqSv,rho)
        # Shstrain=RecStrain(gSh,slowSh,VSh,rho)

        Pdispl=RecDispl(gqP,VqP,rho)
        Svdispl=RecDispl(gqSv,VqSv,rho)
        Shdispl=RecDispl(gSh,VSh,rho)

        P = int(round(TqP/dt))
        Sv = int(round(TqSv/dt))
        Sh = int(round(TSh/dt))
        # strains[P,:]  += Pstrain  * Pprop  * Prad
        # strains[Sv,:] += Svstrain * Svprop * Svrad
        # strains[Sh,:] += Shstrain * Shprop * Shrad


        if P<nsam:
            displs[P,:]  += Pdispl  * Pprop  * Prad
        if Sv<nsam:
            displs[Sv,:] += Svdispl * Svprop * Svrad
        if Sh<nsam:
            displs[Sh,:] += Shdispl * Shprop * Shrad
        
        strt = int(round(t0/dt))
        displwaves=np.apply_along_axis(convolve,0,displs,w,mode='full')[strt:nsam+strt]

        displwaves=displwaves[:nsam]


    return displwaves






def thomsen_2_cij(vp0,vs0,eps,gmm,dlt,rho):
    """
    Function CVTI generates a vti stiffness tensor based on input P and S velocities and
    Thomsen's parameters
   
    Input parameters are:
       vp = P-wave velocity
       vs = S-wave velocity
       eps = Thomsen's epsilon
       gam = Thomsen's gamma
       delt = Thomsen's delta
   
    Calls to other subroutines:
       zeros2d - initialise matrix full of zeros
       symmetry2d - to fill matrix
   
    Written by J.P. Verdon, University of Bristol, 2008-2011
    Converted to Python by A.F. Baird
    """
    c=np.zeros((6,6))
    c[2,2]=vp0*vp0*rho
    c[3,3]=vs0*vs0*rho
    c[0,0]=c[2,2]*(2*eps+1)
    c[5,5]=c[3,3]*(2*gmm+1)     
    c[1,1]=c[0,0]
    c[4,4]=c[3,3]
    c[0,1]=c[0,0]-2*c[5,5] 
    c[0,2]=np.sqrt(2*dlt*c[2,2]*(c[2,2]-c[3,3]) + (c[2,2]-c[3,3])*(c[2,2]-c[3,3])) - c[3,3]     
    if ((2*dlt*c[2,2]*(c[2,2]-c[3,3]) + (c[2,2]-c[3,3])*(c[2,2]-c[3,3]) )< 0):
        raise NameError('HiThere')



    c[1,2]=c[0,2]
    
    # Impose the symmetry (build the lower-left corner).
    for i in range(5):
        for j in range(i,6):
            c[j,i] = c[i,j]
            
    return c
    
    
def fang2VTImt(str,d,r,alpha,C):
    """docstring for fang2VTImt"""
    # CHECK THE RANGE OF THE INPUT ANGLES TO ENSURE THAT UNITARY VECTORS POINT
    # IN THE CORRECT DIRECTION
    if str < 0.:
        str = str + 360.
    if str > 360.:
        str = str - 360.
    if r < -180.:
        r = r + 360.
    if r > 180.:
        r = r - 360.
    if str < 0. or str > 360.:
        raise ValueError('The strike of the fault must be entered in degrees in the range 0 to 360 deg')
    elif d < 0 or d > 90:
        raise ValueError('The dip of the fault must be entered in degrees in the range 0 to 90 deg')
    elif r < -180 or r > 180:
        raise ValueError('The rake of the fault must be entered in degrees in the range -180 to 180 deg')
    elif alpha < -90 or alpha > 90:
        raise ValueError('The angle alpha must be entered in degrees in the range -90 to 90 deg')


    # STIFFNESS TENSOR
    C11 = C[0,0]
    C33 = C[2,2]
    C44 = C[3,3]
    C66 = C[5,5]
    C13 = C[0,2]
    
    # convert degrees to radians
    strR=np.radians(str)
    dR=np.radians(d)
    rR=np.radians(r)
    alphaR=np.radians(alpha)
    

    # NORMAL TO THE FAULT'S PLANE, Vavrycuk 2011, eq. 1 (NOTICE DIFFERENCE IN
    # REFERENCE SYSTEM)
    v = np.array([np.sin(dR)*np.cos(strR),-np.sin(dR)*np.sin(strR),np.cos(dR)])

    # UNITARY SLIP VECTOR, Vavrycuk 2011, eq. 2 (NOTICE DIFFERENCE IN
    # REFERENCE SYSTEM)
    s = np.array([np.cos(rR)*np.sin(strR) - np.cos(dR)*np.sin(rR)*np.cos(strR),np.cos(rR)*np.cos(strR) + np.cos(dR)*np.sin(rR)*np.sin(strR),np.sin(dR)*np.sin(rR)])

    # DISPLACEMENT VECTOR, Vavrycuk 2011, eq. 2
    u = np.cos(alphaR)*s + np.sin(alphaR)*v
    
    
    m=np.zeros(6)

    # MOMENT TENSOR
    m[0] = C11*(v[0]*u[0] + v[1]*u[1]) - 2*C66*v[1]*u[1] + C13*v[2]*u[2]
    m[3] = C11*(v[0]*u[0] + v[1]*u[1]) - 2*C66*v[0]*u[0] + C13*v[2]*u[2]
    m[5] = C13*(v[0]*u[0] + v[1]*u[1]) + C33*v[2]*u[2]
    m[1] = C66*(v[0]*u[1] + v[1]*u[0])
    m[2] = C44*(v[0]*u[2] + v[2]*u[0])
    m[4] = C44*(v[1]*u[2] + v[2]*u[1])

    # NORMALIZATION
    m = m/(np.dot(m.T,m) - 0.5*(m[0]**2+m[3]**2+m[5]**2))**0.5
    
    return u,v,m
    
    
def fang2VTIdt(str,d,r,alpha,C):
    """docstring for fang2VTImt"""
    # CHECK THE RANGE OF THE INPUT ANGLES TO ENSURE THAT UNITARY VECTORS POINT
    # IN THE CORRECT DIRECTION
    if str < 0.:
        str = str + 360.
    if str > 360.:
        str = str - 360.
    if r < -180.:
        r = r + 360.
    if r > 180.:
        r = r - 360.
    if str < 0. or str > 360.:
        raise ValueError('The strike of the fault must be entered in degrees in the range 0 to 360 deg')
    elif d < 0 or d > 90:
        raise ValueError('The dip of the fault must be entered in degrees in the range 0 to 90 deg')
    elif r < -180 or r > 180:
        raise ValueError('The rake of the fault must be entered in degrees in the range -180 to 180 deg')
    elif alpha < -90 or alpha > 90:
        raise ValueError('The angle alpha must be entered in degrees in the range -90 to 90 deg')


    # STIFFNESS TENSOR
    C11 = C[0,0]
    C33 = C[2,2]
    C44 = C[3,3]
    C66 = C[5,5]
    C13 = C[0,2]
    
    # convert degrees to radians
    strR=np.radians(str)
    dR=np.radians(d)
    rR=np.radians(r)
    alphaR=np.radians(alpha)
    

    # NORMAL TO THE FAULT'S PLANE, Vavrycuk 2011, eq. 1 (NOTICE DIFFERENCE IN
    # REFERENCE SYSTEM)
    v = np.array([np.sin(dR)*np.cos(strR),-np.sin(dR)*np.sin(strR),np.cos(dR)])

    # UNITARY SLIP VECTOR, Vavrycuk 2011, eq. 2 (NOTICE DIFFERENCE IN
    # REFERENCE SYSTEM)
    s = np.array([np.cos(rR)*np.sin(strR) - np.cos(dR)*np.sin(rR)*np.cos(strR),np.cos(rR)*np.cos(strR) + np.cos(dR)*np.sin(rR)*np.sin(strR),np.sin(dR)*np.sin(rR)])

    # DISPLACEMENT VECTOR, Vavrycuk 2011, eq. 2
    u = np.cos(alphaR)*s + np.sin(alphaR)*v
    
    
    m=np.zeros(6)

    # MOMENT TENSOR
    m[0] = 2*(v[0]*u[0])
    m[3] = 2*(v[1]*u[1])
    m[5] = 2*(v[2]*u[2])
    m[1] = (v[0]*u[1] + v[1]*u[0])
    m[2] = (v[0]*u[2] + v[2]*u[0])
    m[4] = (v[1]*u[2] + v[2]*u[1])

    # NORMALIZATION
    m = m/(np.dot(m.T,m) - 0.5*(m[0]**2+m[3]**2+m[5]**2))**0.5
    
    return u,v,m
    
    
    
def Brune(fc,dt):
    """docstring for Brune"""
    
    # Seismic moment N.m
    M0 = 1.
    
    # length of pulse
    tlen= 4./fc
    
    # Time axis
    t = np.arange(0,tlen,dt)
    
    # DISPLACEMENT WAVELET
    # w = M0*(2*np.pi*fc)**2*t*np.exp(-2*np.pi*fc*t)

    w = M0*(2*np.pi*fc)**3.*t**2.*np.exp(-2*np.pi*fc*t)/2
  
    return t,w
    
def Beresnev(fc,dt,n=1):
    """docstring for Brune"""
    
    # Seismic moment N.m
    M0 = 1.
    
    # length of pulse
    tlen= 4./fc
    
    # Time axis
    t = np.arange(0,tlen,dt)
    
    om=2*np.pi*fc
    
    # DISPLACEMENT WAVELET
    # w = M0*(2*np.pi*fc)**2*t*np.exp(-2*np.pi*fc*t)

    w = M0* om*(t*om)**n * np.exp(-t*om)/factorial(n)
  
    return t,w
    
    
def gaussian(fc,dt,t0=0):
    """docstring for Brune"""
    
    # Seismic moment N.m
    M0 = 1.
    
    # length of pulse
    tlen= 4./fc+t0
    
    # Time axis
    t = np.arange(0,tlen,dt)
    
    om=2*np.pi*fc
    
    # DISPLACEMENT WAVELET
    # w = M0*(2*np.pi*fc)**2*t*np.exp(-2*np.pi*fc*t)

    w = M0* (om/np.sqrt(2*np.pi))*np.exp(-om**2*(t-t0)**2/2.)
    
    w
  
    return t,w