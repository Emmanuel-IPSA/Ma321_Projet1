######################################################################################
############### Ma321 - Projet1 - Groupe : Grar/Guyon/Khayat/Odenya/Tu ###############
######################################################################################

import numpy as np
import matplotlib.pylab as plt


p = np.loadtxt("dataP.dat") #fichier des âges

q = np.loadtxt("dataQ.dat") #fichier des hauteurs

''' Nous allons commencer par écrire les fonctions, matrices et vecteurs qui
    nous servirons pour l'analyse des résultats numériques'''

#Création du vecteur
def constructX(p):
    X=np.zeros((50,2))
    for i in range(0,50):
        X[i,0]=1
        X[i,1]=p[i]
    return X

#Définition gradient à pas fixe

def gradientpf(A,b,x0,pas,tol,N):
    x = x0
    compteur = 0
    xitpf = []
    while(np.linalg.norm(A@x-b) > tol and N > compteur):
        x = x + pas * (-(A@x - b))
        xitpf.append(x)
        compteur = compteur + 1
        

    return x,compteur,xitpf


#Définition gradient à pas optimal

def gradientPasOptimal(A,b,x0,tol):
    x = x0
    compteur = 0
    xitpo = []
    while(np.linalg.norm(A@x-b) > tol and N > compteur):
        R=A@x-b
        pas = (np.linalg.norm(R)**2/(R.T@A@R))
        x = x + pas * (-(R))
        xitpo.append(x)
        compteur = compteur + 1
        

    return x,compteur, xitpo


#Définition gradient conjugué

def gradientConjugue(A,b,x0,tol):
    x = x0
    compteur = 0
    d= -(A@x-b)
    R=A@x-b
    xitc = []
    while(np.linalg.norm(R) > tol and N > compteur):
        
        pas = (-R.T@d) / (d.T@A@d)
        x = x + pas * d
        R=A@x-b
        d= -R + d* ((R.T@A)@d)/(d.T@A@d)
        compteur = compteur + 1
        xitc.append(x)
    return x,compteur,xitc



if __name__=="__main__":
    

    a = np.ones((50,1))
    p0= np.reshape(p,(50,1))
    X = np.concatenate((a,p0), axis=1) #or X = constructX(p)
   

    #-- gradient pas fixe verification
    A=X.T@X
    b=X.T@q
    x0=np.array([-9,-7]).T
    tol=10**-6
    N=5*10**4
    pas=10**-3 #rho

    #vérificatio pas fixe
    x,nmax, xitpf = gradientpf(A,b,x0,pas,tol,N)
    # print("solution = ",x,"avec ",nmax," itérations")

    
    #-- gradient pas optimal verification
    x1, nmax1, xitpo = gradientPasOptimal(A,b,x0,tol)
    # print("solution = ",x1,"avec ",nmax1," itérations avec gradient pas optimal",xitpo)


     #-- gradient conjugué verification
    x2, nmax2, xitc = gradientConjugue(A,b,x0,tol)
    # print("solution = ",x2,"avec ",nmax2," itérations")

    

#Construction de X.T*X :
XT_X = np.dot(X.T,X)
# print("X.T*X:",XT_X)


#Construction de X.T*q :
XT_q = np.dot(X.T,q)
# print("X.T*q",XT_q)


#Implémentation de la fonction F :
  
norm_q = (np.linalg.norm(q))**2

#Définition fonction F
def fc (c1,c2):
    
    image = (0.5)*((XT_X[0][0])*(c1**2) + 2*(XT_X[0][1])*c1*c2 + (XT_X[1][1])*(c2**2) - 2*(XT_q [0]*c1+XT_q [1]*c2) + norm_q)
    
    return image



### Analyse des résultats numériques

''' Maintenant que nous avons construit les fonctions dont nous aurons besoin,
    nous pouvons passer à l'analyse des résultats'''

## Représentation graphique des méthodes

#Gradient pas fixe

c_etoile=np.array([[0.75016254],[0.06388117]])
plt.figure()
pf1 = list()
pf2 = list()
for i in range (0, len(xitpf)):
    pf1.append(xitpf[i][0])
    pf2.append(xitpf[i][1])
plt.plot(pf1,pf2,color="green", label="méthode du gradient à pas fixe")
plt.ylim(-10,10)
plt.xlim(-10,10)
plt.plot(x0[0],x0[1],'o', color='grey',label='x$_0$')
plt.plot(c_etoile[0],c_etoile[1],'o',color='black', label ='c*', linewidth = 3)
plt.title ('Carte de niveaux avec la méthode du gradient à pas fixe')
plt.legend()
#Affichage des courbes de niveaux
x_carte, y_carte = np.meshgrid(np.linspace(-10,10,201), np.linspace(-10,10,201))
z_carte = fc(x_carte,y_carte)
carte_niveaux = plt.contour(x_carte,y_carte,z_carte,50)
#en dernier show
plt.show()


#Gradient pas optimal
c_etoile=np.array([[0.75016254],[0.06388117]])
plt.figure()
pf3 = list()
pf4 = list()
for i in range (0, len(xitpo)):
    pf3.append(xitpo[i][0])
    pf4.append(xitpo[i][1])
plt.plot(pf3,pf4,color="red", label="méthode du gradient à pas optimal")
plt.ylim(-10,10)
plt.xlim(-10,10)
plt.plot(x0[0],x0[1],'o', color='grey',label='x$_0$')
plt.plot(c_etoile[0],c_etoile[1],'o',color='black', label ='c*', linewidth = 3)
plt.title ('Carte de niveaux avec la méthode du gradient à pas optimal')
plt.legend()
#Affichage des courbes de niveaux
x_carte, y_carte = np.meshgrid(np.linspace(-10,10,201), np.linspace(-10,10,201))
z_carte = fc(x_carte,y_carte)
carte_niveaux = plt.contour(x_carte,y_carte,z_carte,50)
#en dernier show
plt.show()


#Gradient conjugué
c_etoile=np.array([[0.75016254],[0.06388117]])
plt.figure()
pf5 = list()
pf6 = list()
for i in range (0, len(xitc)):
    pf5.append(xitc[i][0])
    pf6.append(xitc[i][1])
plt.plot(pf5,pf6,color="orange", label="méthode du gradient conjugué")
plt.ylim(-10,10)
plt.xlim(-10,10)
plt.plot(x0[0],x0[1],'o', color='grey',label='x$_0$')
plt.plot(c_etoile[0],c_etoile[1],'o',color='black', label ='c*', linewidth = 3)
plt.title ('Carte de niveaux avec la méthode du gradient conjugué')
plt.legend()
#Affichage des courbes de niveaux
x_carte, y_carte = np.meshgrid(np.linspace(-10,10,201), np.linspace(-10,10,201))
z_carte = fc(x_carte,y_carte)
carte_niveaux = plt.contour(x_carte,y_carte,z_carte,50)
#en dernier show
plt.show()




## Visualisation des modèles linéaires

plt.figure()
plt.plot(p, q, "+" , label="Couple ($p_i$,$q_i$)",color ="red") # les points (p, q) representes par des points
  
plt.xlabel("âge des enfants p (année)") # nom de l'axe x
plt.ylabel("Taille des enfant q en (m)") # nom de l'axe y
plt.xlim(1.8, 8.2) # échelle axe x


q_pf = x[0]+p*x[1]
plt.plot(p,q_pf,color='yellow',label='méthode du gradient à pas fixe')

q_po = x1[0]+p*x1[1]
plt.plot(p,q_po,color='cyan',label='méthode du gradient à pas optimal')


q_pc = x2[0]+p*x2[1]
plt.plot(p,q_pc,color='purple',label='méthode du gradient conjugué')


plt.title("Régression linéaire") # titre de graphique
plt.legend()
plt.show()


## Construction du pavé et affichage du gradient

x=np.linspace(-10,10,41)
y=np.linspace(-10,10,41)
C1,C2=np.meshgrid(x,y)

grad = np.gradient(fc (C1,C2))#avec la fonction F définie préalablement
fig_2 = plt.figure()
plt.quiver(C1,C2,grad[0],grad[1])
plt.title("Gradient de la fonction F")
plt.show()




    



