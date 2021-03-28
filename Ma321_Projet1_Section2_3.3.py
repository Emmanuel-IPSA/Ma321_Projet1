######################################################################################
############### Ma321 - Projet1 - Groupe : Grar/Guyon/Khayat/Odenya/Tu ###############
###################################################################################### 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

##### Formulation et analyse mathématique #####

### 2.1.2 Ajustement linéaire

### Question 1. Importation et nuuage de points

p = np.loadtxt("dataP.dat")
q = np.loadtxt("dataQ.dat")

plt.figure(figsize=(16,8))
plt.title("Taille des enfants en fonction de leur âge")
plt.scatter(p, q)
plt.xlabel('Âge des enfants')
plt.ylabel('Taille des enfants')
plt.show()

# Question 5. Calcul de X^TX et X^Tq

One = np.ones(p.size) 
''' On construit tout d'abord le vecteur composé uniquement de 1 
pour pouvoir l'intégrer ensuite aux matrices'''

tXX = np.array([[p.size, p.T@One],[p.T@One,(np.linalg.norm(p))**2]])
''' Nous calculons ici X^T X à partir de son expression''' 

tXq = np.array([q.T@One , p.T@q])
''' Nous calculons ici X^T q à partir de son expression '''



### 2.2.1 Autour de la fonction quadratique F

## Question 1.a. Couples propres de X^TX

m = p.size

X = np.ones((m,2))

''' On construit tout d'abord une matrice (50,2) composée de 1'''

X[:,1] = p

''' On remplace la 2e colonne de la matrice X par le vecteur p'''

A = X.T@X

''' A désigne la matrice (2,2) X^T X
Une autre méthode de construction aurait été d'utiliser
l'expression de X^T X donnée à la question 5'''

cp = np.linalg.eig(A)

Lambda1, v1 = cp[0][0],cp[1][0]/np.linalg.norm(cp[1][0])
Lambda2, v2 = cp[0][1],cp[1][1]/np.linalg.norm(cp[1][1])

n1 = np.linalg.norm(v1)
n2 = np.linalg.norm(v2)
'''On vérifie que les vecteurs propres sont bien unitaires'''


## Question 1.b. Conditionnement

"""Par définition"""
# inv_A = np.linalg.inv(A)
# cond2_A = np.linalg.norm(A) * np.linalg.norm(inv_A)

""" Mais comme A est inversible, on a"""

cond2_A = Lambda2/Lambda1

condn_A = np.linalg.cond(A)

comp = abs(condn_A - cond2_A)/condn_A
''' Calcul de l'erreur relative'''



## Question 3.f. Courbe représentative de de la dérivée partielle

def fonctionF(c):
    return 1/2 * (c.T@tXX)@c - tXq.T@c + 1/2 * (np.linalg.norm(q))**2


def Fpartielle(d, t):
    c_star = np.linalg.inv(tXX)@tXq
    return 1/2 * (d.T@tXX)@d * t**2 + (A@c_star - tXq).T@d * t + fonctionF(c_star)

T = np.arange(-10,11,1)
d_1 = np.array([1, 0])
d_2 = np.array([0, 1])

F1 = np.zeros(len(T)) 
F2 = np.zeros(len(T))
F3 = np.zeros(len(T))
F4 = np.zeros(len(T))

for i in range(len(T)):
    F1[i] = Fpartielle(d_1, T[i])
    F2[i] = Fpartielle(d_2, T[i])
    F3[i] = Fpartielle(v1, T[i])
    F4[i] = Fpartielle(v2, T[i])


plt.title('Fonctions partielles en c* suivant d')
plt.plot(T, F1, color='black', label='avec d = $e_1$')
plt.plot(T, F2, color='orange', label='avec d = $e_2$')
plt.plot(T, F3, color='red', label='avec d = $v_1$')
plt.plot(T, F4, color='green', label='avec d = $v_2$')
plt.axvline(x=0, color='gray', linestyle = '--', label='Axe de symétrie')
plt.legend()
plt.show()



### Question 4.a. Construction des matrices 

X = np.ones((m,2))
X[:,1] = p
''' On construit la matrice X avec une colonne de 1
    et une colonne contenant les données p'''
    
Z = X.T@X

w = X.T@q

s = q.T@q


## Question 4.b. Pavé et courbes de niveau

## Construction du pavé
delta = 0.5
X = np.arange(-10,10.5,delta)
Y = np.arange(-10,10.5,delta)
Xv, Yv = np.meshgrid(X,Y)

## Courbes de niveau
def F(c, A, b, n):
    return 1/2*(A[0,0]*(c[0]**2)+2*A[0,1]*c[0]*c[1]+Z[1,1]*(c[1]**2)-2*(b[0]*c[0]+b[1]*c[1])+n)

matA = np.array([Xv, Yv])
foncF = F(matA, Z, w, s)

fig = plt.figure(figsize=(16,8)) 
plt.title("Courbes de niveau de F")
ax = fig.gca() 
cset = ax.contour(Xv,Yv,foncF,np.arange(0,3000,350))
''' l'ajout de l'argument levels = np.arange( ) nous permet de fixer le nombre de 
    courbes que nous souhaitons avoir'''
plt.colorbar(cset,shrink=0.5,aspect=5)
plt.grid()
plt.show()



## Question 5.a

fig = plt.figure(figsize=(16,8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Xv,Yv,foncF,cmap=cm.coolwarm)
'''Nous reprenons en 3e composante, foncF = F(matA, Z, w, s) calculé précédemment'''
plt.colorbar(surf,shrink=0.5,aspect=5)
ax.set_xlabel('$c_1$')
ax.set_ylabel('$c_2$')
ax.set_zlabel('F')
plt.title("Surface représentative de F")
plt.show()



## Question 8.b. Calcul de c*

Ones = np.ones(m)

Num1 = ((np.linalg.norm(p)**2) * (q.T)@Ones) - ((p.T)@Ones) * ((p.T)@q)
Denom1 = m*(np.linalg.norm(p)**2)-((p.T)@Ones)**2
c_etoile1 = Num1/Denom1

Num2 = (-(p.T)@Ones * (q.T)@Ones) + m*(p.T)@q
Denom2 = (m*(np.linalg.norm(p))**2) - ((p.T)@Ones)**2
c_etoile2 = Num2/Denom2

c_etoile = np.array([c_etoile1, c_etoile2])
c_star = np.linalg.solve(Z, w)

#Comparaison

ecart = abs((np.linalg.norm(c_star)-np.linalg.norm(c_etoile)))/np.linalg.norm(c_star)

### Plus simplement, on a
    
# c1 = (((np.linalg.norm(p)**2) * (q.T)@Ones) - ((p.T)@Ones) * ((p.T)@q))/(m*(np.linalg.norm(p)**2)-((p.T)@Ones)**2)
# c2 = ((-(p.T)@Ones * (q.T)@Ones) + m*(p.T)@q)/((m*(np.linalg.norm(p))**2) - ((p.T)@Ones)**2)
# c_etoile =(c1, c2)




##### Méthodes à directions de descente : une étude numérique #####

### 3.2.2 L'algorithme de descente du gradient à pas fixe

x0= np.array([-9,-7])

tol=10**(-6)
rho=10**(-3)

## Question 4. Gradient à pas fixe

def GradientPasFixe(A,b,x0,rho,tol):
    xit = []
    xit.append(x0)
    ''' Nous fixons ici une valeur initale pour la solution'''
    i = 1
    iMax = 5*10**4
    r = 1
    x = x0
    while (np.linalg.norm(r)>tol and i<iMax):
        r = A@x-b
        ''' Nous calculons à chaque itération le résidu 
        du problème'''
        d = -r
        ''' Grâce au résidu, nous déterminons la direction 
        de descente'''
        x = x+rho*d
        ''' On met à jour x en nous déplaçant dans la
        direction de descente'''
        i += 1
        xit.append(x)
        ''' Cette commande nous permet de stocker tous
        les xi calculés'''
    return(x,xit,i)


Res1= GradientPasFixe(Z,w,x0,rho,tol)

print('\n Pour tol = 1e-6, le gradient à pas fixe donne c* =', Res1[0], 'au bout de', Res1[2], 'itérations')


##  Test  avec rho = 1e-1 et rho = 1e-5

#Res11 = GradientPasFixe(Z,w,x0,1e-1,tol)
#Res12 = GradientPasFixe(Z,w,x0,1e-5,tol)
''' Décommenter pour afficher les résultats '''


### 3.2.3 L'algorithme de descente du gradient à pas optimal

## Question 6. Gradient à pas optimal

def GradientPasOptimal(A,b,x0,tol):
    xit = []
    xit.append(x0)
    ''' Initialisation à partir de la condition initale x0 = c0'''
    i = 1
    iMax = 5*10**4
    r = 1
    x = x0
    while (np.linalg.norm(r)>tol and i<iMax):
        r = A@x-b
        d = -r
        rho= (np.linalg.norm(r)**2)/(np.transpose(r)@A@r)
        ''' On adapte le pas en fonction du résidu '''
        x = x+rho*d
        i += 1
        xit.append(x)
    return(x,xit,i)

    
Res2= GradientPasOptimal(Z,w,x0,tol)

print('\n Pour tol = 1e-6, le gradient à pas optimal donne c* =', Res2[0], 'au bout de', Res2[2], 'itérations')



### 3.3 L'algorithme des gradients conjugués

## Question 5. Gradient conjugué

def gradientConjugue(A,b,x0,tol):
    x = x0
    compteur = 0
    iMax = 5*10**4
    d= -(A@x-b)
    R=A@x-b
    xit = []
    while(np.linalg.norm(R) > tol and iMax > compteur):
        
        pas = (-R.T@d) / (d.T@A@d)
        x = x + pas * d
        R=A@x-b
        d= -R + d* ((R.T@A)@d)/(d.T@A@d)
        compteur = compteur + 1
        xit.append(x)
    return (x,xit,compteur)

Res3 = gradientConjugue(Z,w,x0,tol)

print('\n Pour tol = 1e-6, la méthode du gradient conjugué donne c* =', Res3[0], 'au bout de', Res3[2], 'itérations')





