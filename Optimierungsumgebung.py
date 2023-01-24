import time
import numpy as np
import nlopt_wrapper


def Optimierungsfunktion(Faktoren,a,b):
    #Zielvektor ist Funktion f = a+b*x+c*x² mit a=0.5,b=-0.8,c=1.2
    Zielvektor = np.array([[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10],[128.5,104.9,83.7,64.9,48.5,34.5,22.9,13.7,6.9,2.5,0.5,0.9,3.7,8.9,16.5,26.5,38.9,53.7,70.9,90.5,112.5]],dtype=np.float64)

    #Faktoren in Variablen aufteilen (Anschaulicher)
    a=Faktoren[0]
    b=Faktoren[1]
    c=Faktoren[2]
    print(Faktoren)

    #Ergebnis Vektor definieren und berechnen mit den Faktoren
    Ergebnis = np.ones((2,21))
    Ergebnis[0]=Zielvektor[0]
    Ergebnis[1]=a+b*Ergebnis[0]+c*(Ergebnis[0]**2)

    #Zielfunktion ist die Durchschnittliche Abweichung oder ähnliches, meisten Minimierung der kleinsten Quadrate
    DurchschnittlicheAbweichung=(Ergebnis-Zielvektor)**2
    print(np.sum(DurchschnittlicheAbweichung))

    #Faktor zur Verlangsamung in Sekunden, da sonst nicht anschaulich. Entfernen wenns schnell gehen soll...
    #time.sleep(0.1)

    #Rückgabewert an den Optimierer
    return np.sum(DurchschnittlicheAbweichung)

def Opt():
    global bnds;
    global mittl_Abweichung
    #Obere und untere Grenzen für den Optimierer beim Verändern der Variablen
    ubnds = ((50), (50), (50))
    lbnds = ((-50), (-50), (-50))

    #ubnds = ((10), (50), (50))
    #lbnds = ((-10), (-50), (-50))

    #Startwerte für den Optimierer
    Startwerte =[1.1,1.1,1.1]
    #Eigentlicher Optimiereraufruf mit verschiedenen Algorythmen möglich, siehe nlopt

    #NLopt derivative-free optimizers are:
    #______________________________________________________
    Optergebnis = nlopt_wrapper.BOBYQA(Optimierungsfunktion, x0=Startwerte, args=(0, 0), ubnds=ubnds, lbnds=lbnds,tol=0.00001, xtol=0.001)
    #Optergebnis = nlopt_wrapper.subplex(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.AUGLAG(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.AUGLAG_EQ(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.COBYLA(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.NEWUOA(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.NEWUOA_BOUND(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.PRAXIS(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.NELDERMEAD(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)

    #NLopt gradient-based optimizers are:
    # zur Zeit nicht richtig implementiert, da kein Gradient vorgegeben...
    #Optergebnis = nlopt_wrapper.LD_SLSQP(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.LD_LBFGS(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.LD_SLSQP(Optimierungsfunktion, GalacticOptim.AutoForwardDiff())

    # NLopt optimizers with constraint equations:
    # ______________________________________________________
    #Optergebnis = nlopt_wrapper.GN_ISRES(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, maxtime=10000000, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.GN_ORIG_DIRECT_L(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, maxtime=100000, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.GN_AGS(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, maxtime=100000, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)

    # NLopt optimizers w/o constraint equations:
    # ______________________________________________________
    # Optergebnis = nlopt_wrapper.GN_DIRECT(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, maxtime=100000, lbnds=lbnds, args=(0, 0), tol=0.000001, xtol=0.00001)
    #Optergebnis = nlopt_wrapper.GN_CRS2_LM(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, maxtime=100000, lbnds=lbnds, args=(0, 0), tol=0.0000001, xtol=0.000001)
    #Optergebnis = nlopt_wrapper.GD_STOGO_RAND(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, maxtime=100000, lbnds=lbnds, args=(0, 0), tol=0.0000001, xtol=0.000001)
    #Optergebnis = nlopt_wrapper.GN_MLSL(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, maxtime=100000, lbnds=lbnds, args=(0, 0), tol=0.0000001, xtol=0.000001)
    #Optergebnis = nlopt_wrapper.GN_MLSL_LDS(Optimierungsfunktion, x0=Startwerte, ubnds=ubnds, maxtime=50000, lbnds=lbnds, args=(0, 0), tol=0.0000001, xtol=0.000001)

    try:
        print(x0)
    except:
        print("no x0\n Ende der Optimierung.....")
    bres = np.around(Optergebnis.x, 2)
    #print(bres)
    print("Faktor ist a:",bres[0]," ,b=",bres[1]," und c=",bres[2])
    #print(Optimierungsfunktion(bres,1,2))

# Schalter für die manuelle Nachrechnung der Funktion mit eigenen Faktoren
Schalter = 1

if Schalter == 1:
    Opt()
else:
    Startwerte =[0.5,-0.8,1.2]
    Optimierungsfunktion(Startwerte,1,1)