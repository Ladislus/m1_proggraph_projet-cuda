# Projet CUDA
**Membres du groupe:**  
 - Tom RIBARDIÈRE (2171029)  
 - Ladislas WALCAK (2174867)

## Algorithmes implantés

Nous avons implémenté 2 algoritmes différents :

### Algorithme Grayscale

Lors du développement de l'effet ASCII Art, nous avons utilisé une fonction OpenCV permettant l'ouverture de l'image directement en niveau de gris 
(fonction `cv::imread` avec le flag `cv::IMREAD_GRAYSCALE`). Ne sachant pas si cela été autorisé, nous avons implémenté cet algorithme en CPU et GPU afin de prouver
que nous étions capable de la réaliser, mais ne l'avons utilisé dans les autres effets. A l'ouverture, nous utilisons une fonction de la librairie OpenCV qui nous permet de ne récupérer l'image en 3 channels de couleurs, nous permettant donc de prendre en entrée des images de 3 channels ou plus. Ce code applique à chaque channels de couleur de tous les pixels de 
l'image (rouge, vert, bleu), des constantes multiplicatives qui nous permettent d'obtenir une nuance de gris en fonction de l'intensité des couleurs d'un pixel.

### Algorithme ASCII Art

Pour cet algorithme, nous regardons l'intensité des pixels de l'image d'entrée qui est importée en nuance de gris avec une fonction OpenCV.
Lors du traitement, nous faisons une correspondance entre l'intensité d'un pixel de l'image et une liste de caractère pour choisir le caractère qui sera le plus adapté à l'intensité.
Après avoir choisi le bon caractère dans la liste, nous l'ajoutons dans un fichier texte. Lors de l'écriture du fichier de sorti, nous écrivons chaque caractère en 3 exemplaires, 
afin de palier à la déformation de l'image (les caractères étant plus hauts que larges, les images ont tendance à être aplatie).

### Algorithme Convolution

Cet algorithme a pour but de modifier un pixel en fonction de ces voisins.

Pour cet effets, les images doivent obligatoirement être en 4 channels. Si ce n'est pas le cas, une exception `std::logic_error` sera levée avec un message d'erreur vous avertissant que
votre image n'est pas conforme. Lors du traitement, nous parcourons l'image, pixels par pixel, en prenant soin d'éviter les bordures de l'image car les pixels de ces bordures n'ont pas 8 voisins.
Pour obtenir un effet, il nous suffit de choisir une matrice appelée *noyau* ou *kernel*, qui va nous indiquer comment changer la valeur de notre pixel en fonction de ces voisins.
Il faut également que vous choisissiez un deviseur (division de la somme totale finale) et un décalage (necessaire dans les cas ou la somme, après division, pourrait être supérieure à 255)
qui feront parti de la formule appliquée pour le changement de valeur des pixels.

Vu que cet effet est assez générique et que nous n'avons que le kernel (matrice), le diviseur et le décalage qui sont des variables, nous pouvons ajouter des effets assez facilement (comme par exemple plusieurs types de flou, plusieurs type de detections de contours, des rotations, l'amélioration de la netteté, augmentation de contraste, des redimensions, effets mirroir, déformations ...).
Le code contient une map permettant d'associer un fonction à une enum, afin d'appliquer l'effet désiré. Ces fonctions ne sont la que pour changer les paramètres initiaux.  
Le code CPU actuel contient les effets :
- Flou Box
- Flou Gaussien
- Détéction de bord

Pour le code GPU, nous avons seulement mis les paramètres initiaux de la détéction de bord, cependant, pour avoir les 2 autres effets, il vous suffit de changer le kernel, le diviseur et le décalage dans le code. 

Nous avons identifié un problème sur cet algoritme pour la détéction de bord, entre les versions CPU et GPU. Nous avons, pour la version CPU, des bords de couleurs orange et rouge qui ne semble pas à leurs places, et pour la version GPU tout est presque seulement composé de bleu turquoise. Nous ne savons pas quelle implémentation est la bonne, et n'avons pas réussi à trouver la cause de ce problème (nous soupçonnons que la fonction `cv::Mat.at()` retourne le pixel en RGB, mais au sein de la matrice, les données sont au format BGR).

## Optimisations choisies

Lors de la phase d'optimisation du code, nous n'avons pas trouvé beaucoup d'améliorations à apporter.

N'ayant aucune communication entre les threads, la mémoire shared et les streams ne sont pas utiles. Nous avons essayé d'utiliser au plus les constantes sur le HOST et le DEVICE, et
avons utilisé au maximum les pointers pour éviter la copie de structure de données complexes.  

Pour ce qui est des boucles et des conditionnelles, les codes GPU ne contiennent que très peu de boucles, et la plupart des conditionnels n'ont pas de partie "else", ne doublant donc pas le
temps d'éxecution de ces blocks de code. 

Pour l'optimisation des tailles de blocks, nous avons réalisé des tests afin de déterminer le nombre de threads optimial pour chacun des codes.

## Difficultés rencontrées

1. Trouver des images correctes
   Lors du développement, nous avons quelques difficultés à trouver des images correctes. Dû au fait que le programme Convolution ait besoin d'images en 4 channels, les images devaient être
   au format PNG, et être d'une taille suffisante pour que les tests soient concluant. Cependant, les images ne pouvaient pas être de trop grande taille, car cela devenait difficile de les envoyer
   sur le serveur de test. De plus, il est également arrivé que des images puissent être ouvert, mais soit malformée lors de l'ouverture avec OpenCV (Checksum CRC incorrect, ...).  
   
2. Utilisation de la machine distante 
   Afin de tester les programmes GPU, nous devions faire beaucoup les modifications sur nos machines personnelles, puis push ces modifications sur le dépôt Git. Cela a eu pour effet de remplir le dépôt
   avec des commits non fonctionnel, et rendant toutes les modifications, même les plus simple, assez compliqué.  
   
3. Ascii art au format texte
   Lors du développement de l'effet ASCII art, nous n'avons pas trouver de moyen simple de générer un image affichant les caractères ASCII désirés. Nous avons donc
   fait en sorte que le programme génère un fichier de sortie au format texte.  
   
## Tests réalisés 

Après avoir réalisé les différents codes dans leurs 2 versions (CPU et GPU), nous nous sommes mis sur l'étude de temps.
Voici un lien vers le tableur dans lequel nous avons mis nos relevés : [Fichier local](resources/benchmarks.pdf), [Lien tableur](https://docs.google.com/spreadsheets/d/1R7NSRRONeWaLxqyIPqeSVBCTHMc6pXrAg4N14QCkn4A/edit?usp=sharing)  

Tous les tests des versions GPU des programmes ont été réalisé sur la machine distante fournie, dû au fait qu'aucun membre du groupe n'avait de carte graphique NVidia supportant le CUDA à disposition.

Afin d'effectuer ces tests, nous avons écrit un [petit script bash](test.sh) permettant de lancer un test pour chacun des 6 codes. Le temps d'éxecution est recupéré grâce à la commande bash `time`, 
qui est certe très inprécise, mais qui permet d'obtenir une bonne idée de l'ordre de temps d'éxécution. Tous les tests sont effectués sur l'image [Celeste](input/Celeste.png).  

Lors des tests, nous avons eu plusieurs problèmes quand au format des images, c'est pour cela que nous avons mis en place un certain nombre de restriction dans les programmes, 
permettant d'éviter tous ces problèmes, notamment pour l'effect Convolution, ou nous obligeons l'image d'entrée à être en 4 channels.

Nous pouvons voir que pour les 3 effets, plus nous avons de threads, plus le temps des évents CUDA, c'est à dire le temps de calcul "réel", est court. Cependant, la plupart du temps, ce gain est
perdu, dans un premier temps lors de la communication des images au device, mais aussi par le fait qu'un plus grand nombre de threads nécessite plus de temps d'initialisation/création des threads.

### Grayscale
Nous pouvons observer que sur l'effet Grayscale, nous n'avons pas ou presque pas de différence de temps total sur les éxécutions CPU et
GPU (différence 0,046s au plus). Cependant, nous pouvons voir que pour le code GPU, le temps de traitement "réel" (colonne "event") ne correspond qu'a une infime partie du temps total, 
montrant bien que le traitement est extremement rapide, mais que les temps de transferts sont conséquents.

### ASCII Art
Nous pouvons avoir les mêmes observations que pour l'effet grayscale, nous avons peu de différence de temps réel entre les éxécutions
CPU et GPU (différence 0,087s au plus).

### Convolution
Pour cet effet, nous pouvons observer une "réelle" différence de temps entre les exécutions CPU et GPU (différence de 1,515s au plus).
Nous avons donc un réel "gain de temps" en utilisant la version GPU contrairement à la version CPU.













